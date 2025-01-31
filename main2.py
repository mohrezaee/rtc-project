
import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import deque

# ================================================================
#                  PARAMETER SETTINGS
# ================================================================
NUM_TASKS = 10
NODES_RANGE = (3, 6)
WCET_RANGE = (5, 10)
P_EDGE = 0.2
D_RATIO_RANGE = (0.2, 0.4)
RESOURCE_RANGE = (1, 3)
ACCESS_COUNT_RANGE = (1, 4)
ACCESS_LEN_RANGE = (2, 5)
CRITICAL_RATIOS = [1, 0.5]
OUTPUT_DIR = "graphs_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================
#                 HELPER / UTILITY FUNCTIONS
# ================================================================

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """ Create a hierarchical layout for a tree-like DAG. """
    pos = {root: (xcenter, vert_loc)}
    neighbors = list(G.successors(root))
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos.update(hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx))
    return pos

def compute_is_on_critical_path(G, node_types, source_id, sink_id):
    """
    Marks each node in G as on a 'critical path' if it lies on some s->...->t path
    that includes at least one Hard node.
    """
    # 1) Find all nodes reachable from source
    reachable_from_source = set()
    queue = [source_id]
    while queue:
        u = queue.pop()
        for v in G.successors(u):
            if v not in reachable_from_source:
                reachable_from_source.add(v)
                queue.append(v)

    # 2) Find all nodes that can reach sink (backward)
    can_reach_sink = set()
    G_rev = G.reverse()
    queue = [sink_id]
    while queue:
        u = queue.pop()
        for v in G_rev.successors(u):
            if v not in can_reach_sink:
                can_reach_sink.add(v)
                queue.append(v)

    # ST = intersection
    ST = set()
    for v in G.nodes():
        if v in reachable_from_source and v in can_reach_sink:
            ST.add(v)

    # Hard nodes in ST
    hard_nodes = [v for v in ST if node_types.get(v, None) == "Hard"]
    if not hard_nodes:
        return {v: False for v in G.nodes()}

    critical_set = set()

    def bfs_forward(start):
        q = [start]
        visited = {start}
        while q:
            u = q.pop(0)
            for nxt in G.successors(u):
                if nxt in ST and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return visited

    def bfs_backward(start):
        q = [start]
        visited = {start}
        while q:
            u = q.pop(0)
            for nxt in G_rev.successors(u):
                if nxt in ST and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        return visited

    for h in hard_nodes:
        fwd = bfs_forward(h)
        bwd = bfs_backward(h)
        critical_set.update(fwd)
        critical_set.update(bwd)

    is_on_crit = {}
    for v in G.nodes():
        is_on_crit[v] = (v in ST) and (v in critical_set)
    return is_on_crit

def compute_longest_path_length_dag(G, wcet):
    """
    Computes the length of the longest path in the DAG using each node's wcet.
    """
    topo_order = list(nx.topological_sort(G))
    dp = {node: wcet[node] for node in G.nodes()}
    for u in topo_order:
        for v in G.successors(u):
            if dp[v] < dp[u] + wcet[v]:
                dp[v] = dp[u] + wcet[v]
    return max(dp.values()) if dp else 0

# ================================================================
#                OBJECT-ORIENTED CLASSES
# ================================================================

class ResourceInterval:
    """
    Represents a time interval [start, end) within the node's execution
    during which it needs a particular resource_id.
    For example: resource=1 from time_executed=[3..5).
    """
    def __init__(self, start, end, resource_id):
        self.start = start
        self.end = end
        self.resource_id = resource_id

    def __repr__(self):
        return f"(Res={self.resource_id}, [{self.start}..{self.end}))"


class Node:
    """
    Represents a single node in the DAG.
    - Stores partial resource usage intervals in resource_intervals.
    """
    def __init__(self, node_id, wcet, is_hard=False, on_critical_path=False):
        self.node_id = node_id
        self.wcet = wcet
        self.is_hard = is_hard
        self.on_critical_path = on_critical_path

        # For partial usage:
        self.resource_intervals = []  # list of ResourceInterval
        self.current_exec_progress = 0  # how many time units executed so far

        # Execution state:
        self.remaining_time = wcet
        self.in_critical_section = False
        self.completed = False
        self.deadline = None  # will be assigned by the Task

        # Track which resource we currently hold (if in a critical section)
        self.locked_resource = None

    def add_resource_interval(self, start, end, resource_id):
        """
        Add an interval [start, end) for resource usage.
        Must have 0 <= start < end <= wcet.
        """
        interval = ResourceInterval(start, end, resource_id)
        self.resource_intervals.append(interval)

    def get_resource_needed_now(self):
        """
        Returns the resource_id this node needs at the
        current_exec_progress (if any), else None.
        This checks if 'current_exec_progress' is within [start..end).
        """
        for interval in self.resource_intervals:
            if interval.start <= self.current_exec_progress < interval.end:
                return interval.resource_id
        return None

    def reset_execution_state(self):
        self.remaining_time = self.wcet
        self.current_exec_progress = 0
        self.in_critical_section = False
        self.completed = False
        self.locked_resource = None

    def __repr__(self):
        return (f"Node(id={self.node_id}, wcet={self.wcet}, is_hard={self.is_hard}, "
                f"res_intervals={self.resource_intervals})")


class Task:
    """
    Represents a single task, which is a DAG of Node objects + source/sink.
    Also stores the usual real-time parameters (C, L, D, T, U).
    """
    def __init__(self, task_id, G, node_list, source_id, sink_id):
        self.task_id = task_id
        self.G = G
        self.nodes = node_list  # list[Node]
        self.source_id = source_id
        self.sink_id = sink_id

        # Real-time parameters
        self.C = 0
        self.L = 0
        self.D = 0
        self.T = 0
        self.U = 0
        self.critical_ratio = 0

    def compute_params(self):
        """Compute C, L, D, T, U from the node data and the DAG."""
        self.C = sum(n.wcet for n in self.nodes
                     if n.node_id not in [self.source_id, self.sink_id])

        wcet_map = {n.node_id: n.wcet for n in self.nodes}
        self.L = compute_longest_path_length_dag(self.G, wcet_map)

        ratio_d = random.uniform(*D_RATIO_RANGE)
        self.D = int(self.L / ratio_d) if ratio_d != 0 else self.L
        self.T = self.D
        self.U = self.C / self.T if self.T > 0 else float('inf')

    def set_deadlines_for_nodes(self):
        """Assign each node the same deadline = self.D (only for demonstration)."""
        for n in self.nodes:
            n.deadline = self.D

    def __repr__(self):
        return (f"Task(id={self.task_id}, C={self.C}, L={self.L}, "
                f"D={self.D}, U={self.U:.2f}, #nodes={len(self.nodes)})")


class Resource:
    """
    Represents a shared resource in the system.
    - Maintains a 'global queue' for POMIP.
    """
    def __init__(self, resource_id, max_access_length):
        self.resource_id = resource_id
        self.max_access_length = max_access_length
        self.global_queue = deque()  # (node, arrival_time, is_crit_path, is_hard)

    def enqueue_global(self, item, policy="FIFO"):
        self.global_queue.append(item)
        if policy == "FPC":
            self._sort_by_critical_path()

    def _sort_by_critical_path(self):
        def priority(elem):
            # Node is critical path => higher priority (sort ascending by NOT on_critical_path)
            # tie-break arrival_time
            return (not elem["on_critical_path"], elem["arrival_time"])
        sorted_list = sorted(self.global_queue, key=priority)
        self.global_queue = deque(sorted_list)

    def pop_front(self):
        if self.global_queue:
            return self.global_queue.popleft()
        return None

    def remove_item(self, node_obj):
        """
        Remove 'node_obj' from the global queue if present.
        """
        for i, it in enumerate(self.global_queue):
            if it["node"] == node_obj:
                del self.global_queue[i]
                break

    def front(self):
        return self.global_queue[0] if self.global_queue else None

    def __repr__(self):
        return f"Resource(id={self.resource_id}, max_len={self.max_access_length})"

# For a simple single-processor model:
class Processor:
    """
    Represents a single CPU/Processor that can run a single Node at a time.
    """
    def __init__(self, proc_id):
        self.proc_id = proc_id
        self.occupant = None  # Node object or None

    def is_free(self):
        return (self.occupant is None)

    def assign(self, node_obj):
        self.occupant = node_obj

    def release(self):
        self.occupant = None

    def __repr__(self):
        return f"Processor(id={self.proc_id}, occupant={self.occupant})"


# ================================================================
#               DAG + TASK GENERATION CODE
# ================================================================

def create_random_dag(num_inner, p):
    """
    Creates a random DAG with num_inner intermediate nodes,
    plus source (id=num_inner) and sink (id=num_inner+1).
    Returns (G, node_types, ratio_crit).
    """
    total_nodes = num_inner + 2
    source_id = num_inner
    sink_id = num_inner + 1

    while True:
        G = nx.DiGraph()
        G.add_nodes_from(range(total_nodes))

        node_types = {n: None for n in range(total_nodes)}
        for n in range(num_inner):
            node_types[n] = "Soft"

        ratio_crit = random.choice(CRITICAL_RATIOS)
        ratio_crit /= (ratio_crit + 1)
        num_crit = round(ratio_crit * num_inner)
        crit_nodes = set(random.sample(range(num_inner), num_crit)) if num_inner > 0 else set()
        for n in crit_nodes:
            node_types[n] = "Hard"

        # Add edges
        for u in range(num_inner):
            for v in range(u+1, num_inner):
                if random.random() < p and not (node_types[u] == "Soft" and node_types[v] == "Hard"):
                    G.add_edge(u, v)

        if num_inner > 0:
            G.add_edge(source_id, 0)
            G.add_edge(num_inner-1, sink_id)
            for n in range(num_inner):
                if not nx.has_path(G, source_id, n):
                    G.add_edge(source_id, n)

        try:
            nx.topological_sort(G)
            return G, node_types, ratio_crit
        except nx.NetworkXUnfeasible:
            continue

def generate_one_task(task_id):
    """
    Create a Task object with Node objects, connect them in a DAG, etc.
    """
    num_inner = random.randint(*NODES_RANGE)
    G_mid, node_types, ratio_crit = create_random_dag(num_inner, P_EDGE)
    source_id = num_inner
    sink_id = num_inner + 1
    total_nodes = num_inner + 2

    G = nx.DiGraph()
    G.add_nodes_from(range(total_nodes))
    G.add_edges_from(G_mid.edges())
    if num_inner > 0:
        G.add_edge(source_id, 0)
        G.add_edge(num_inner-1, sink_id)

    # Create Node objects
    node_list = []
    for n in range(total_nodes):
        if n in (source_id, sink_id):
            w = 0
        else:
            w = random.randint(*WCET_RANGE)
        is_hard = (node_types[n] == "Hard")
        node_list.append(Node(node_id=n, wcet=w, is_hard=is_hard))

    # Mark on_critical_path
    _ntypes = {n: node_types[n] for n in range(total_nodes)}
    is_crit_map = compute_is_on_critical_path(G, _ntypes, source_id, sink_id)
    for nd in node_list:
        nd.on_critical_path = is_crit_map[nd.node_id]

    # Build Task
    t = Task(task_id, G, node_list, source_id, sink_id)
    t.critical_ratio = ratio_crit
    t.compute_params()
    t.set_deadlines_for_nodes()

    # Visualization
    pos = hierarchy_pos(t.G, source_id)
    plt.figure(figsize=(5, 4))
    node_colors = []
    for nd in t.nodes:
        if nd.node_id in (source_id, sink_id):
            node_colors.append("yellow")
        elif nd.is_hard:
            node_colors.append("red")
        else:
            node_colors.append("green")

    nx.draw_networkx_nodes(t.G, pos,
                           nodelist=[nd.node_id for nd in t.nodes],
                           node_color=node_colors,
                           node_size=500, alpha=0.9)
    nx.draw_networkx_edges(t.G, pos, arrows=True)
    labels_dict = {nd.node_id: str(nd.node_id) for nd in t.nodes}
    nx.draw_networkx_labels(t.G, pos, labels=labels_dict, font_color='white')
    plt.title(f"Task {task_id} | Ci={t.C} | Li={t.L} | Di={t.D}")
    plt.axis('off')
    png_path = os.path.join(OUTPUT_DIR, f"dag_task_{t.task_id}.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    return t


def generate_resources(num_tasks):
    """
    Return a list of Resource plus metadata for printing.
    """
    n_r = random.randint(*RESOURCE_RANGE)
    resources = []
    info_list = []
    for r_id in range(n_r):
        max_len = random.randint(*ACCESS_LEN_RANGE)
        total_acc = random.randint(*ACCESS_COUNT_RANGE)
        distribution = [0]*num_tasks
        remain = total_acc
        for i in range(num_tasks - 1):
            if remain <= 0: break
            pick = random.randint(0, remain)
            distribution[i] = pick
            remain -= pick
        distribution[-1] += remain

        r_obj = Resource(resource_id=r_id+1, max_access_length=max_len)
        resources.append(r_obj)
        info_list.append({
            "resource_id": r_id + 1,
            "max_access_length": max_len,
            "total_accesses": total_acc,
            "distribution": distribution
        })

    return resources, info_list

# ================================================================
#         Assign Resource Usage Intervals to Each Task
# ================================================================

def assign_resource_intervals_to_tasks(tasks, resources, resource_info):
    """
    We interpret resource_info[r]["distribution"][task_idx] as the
    total number of resource usages for resource r by task idx.
    We then distribute these usage *intervals* among the nodes of the task.

    - We pick nodes that have nonzero wcet.
    - For each usage, we pick a random node, pick an interval length within
      the resource's max_access_length, then pick a random start time in [0..wcet-interval_length].
    - Add a ResourceInterval to that node.
    """
    # Build a map: resource_id -> resource object
    rmap = {r.resource_id: r for r in resources}

    for i, t in enumerate(tasks):
        # for each resource, see how many usages this task has
        for rinfo in resource_info:
            rid = rinfo["resource_id"]
            usage_count = rinfo["distribution"][i]  # how many intervals
            max_len = rinfo["max_access_length"]
            # get all nodes that have wcet>0 (excluding source, sink)
            candidate_nodes = [nd for nd in t.nodes if nd.wcet > 0]
            for _ in range(usage_count):
                if not candidate_nodes:
                    break
                nd = random.choice(candidate_nodes)
                # pick interval length <= max_len but also <= nd.wcet
                interval_length = random.randint(1, min(max_len, nd.wcet))
                start = random.randint(0, nd.wcet - interval_length)
                end = start + interval_length
                nd.add_resource_interval(start, end, rid)
    # Now each node may have multiple intervals from possibly different resources

# ================================================================
#               FEDERATED SCHEDULING UTILS
# ================================================================

def compute_processors_federated(tasks):
    total = 0
    for t in tasks:
        if t.U > 1:
            denom = (t.D - t.L) if (t.D - t.L) > 0 else 1
            top = (t.C - t.L)
            m_i = math.ceil(top / denom)
            if m_i < 1: m_i = 1
            total += m_i
        else:
            total += 1
    return total

def federated_output(tasks, resource_info):
    U_sum = sum(t.U for t in tasks)
    U_norm = random.uniform(0.1, 1)
    m_simple = math.ceil(U_sum/U_norm)
    m_fed = compute_processors_federated(tasks)

    print("==================================================")
    print(" Task and Resource Generation (Phase One) ")
    print("==================================================")
    for t in tasks:
        print(f"\n--- Task tau_{t.task_id} ---")
        print(f" • #nodes={len(t.nodes)}")
        print(f" • RatioCrit={t.critical_ratio:.2f}")
        print(f" • C={t.C}, L={t.L}, D={t.D}, U={t.U:.2f}")

    print("--------------------------------------------------")
    print(f" • Total Utilization (UΣ) = {U_sum:.3f}")
    print("--------------------------------------------------")

    print("\n==================================================")
    print(" Shared Resources ")
    print("==================================================")
    print(f" • # of resources = {len(resource_info)}")
    for r in resource_info:
        dist_str = ", ".join([f"tau_{idx + 1}={val}" for idx, val in enumerate(r['distribution'])])
        print(f"  - Res l_{r['resource_id']} maxLen={r['max_access_length']}, totalAcc={r['total_accesses']}")
        print(f"    Distribution: {dist_str}")

    print("\n==================================================")
    print(" Required Processors ")
    print("==================================================")
    print(f" • Simple approach: m = {m_simple}")
    print(f" • Federated approach: m = {m_fed}")
    print("==================================================\n")
    return m_fed, m_simple

# ================================================================
#           POMIP-LIKE SCHEDULER WITH PARTIAL USAGE
# ================================================================

def schedule_with_pomip(tasks, resources, time_limit=2000, policy="FIFO"):
    """
    Time-stepped, 1 CPU per task (federated).
    Checks:
    - Precedence constraints: a node can only run if all preds are done
    - Partial resource usage: request resource when current_exec_progress in [start..end) for that resource
    """
    # 1 CPU per task
    processors = [Processor(i) for i in range(len(tasks))]
    resource_map = {r.resource_id: r for r in resources}

    # Flatten nodes for easy lookup: (task_id, node_id) -> Node
    node_lookup = {}
    for t in tasks:
        for nd in t.nodes:
            node_lookup[(t.task_id, nd.node_id)] = nd

    # Helper: check if node is "runnable" => not completed, all preds done
    def is_runnable(t: Task, nd: Node):
        if nd.completed: return False
        # check all preds
        for pred in t.G.predecessors(nd.node_id):
            pred_node = node_lookup[(t.task_id, pred)]
            if not pred_node.completed:
                return False
        return True

    # Helper: pick a node from task i, priority to Hard if both Hard & Soft are available
    def pick_node_to_run(task_index):
        t = tasks[task_index]
        cands = []
        for nd in t.nodes:
            if not nd.completed and is_runnable(t, nd):
                cands.append(nd)

        # Hard first
        c_hard = [c for c in cands if c.is_hard and not c.in_critical_section]
        c_soft = [c for c in cands if not c.is_hard]
        if c_hard:
            return c_hard[0]
        elif c_soft:
            return c_soft[0]
        return None

    time = 0
    schedulable = True

    while time < time_limit:
        # if all nodes done
        alldone = True
        for nd in node_lookup.values():
            if not nd.completed and nd.wcet > 0:
                alldone = False
                break
        if alldone: 
            break

        # For each processor
        for p in processors:
            if p.is_free():
                # pick node
                nd = pick_node_to_run(p.proc_id)
                if nd:
                    p.assign(nd)
            else:
                # occupant is running
                nd = p.occupant
                # check if we need resource now
                needed_res_id = nd.get_resource_needed_now()  # checks intervals
                if needed_res_id is not None and not nd.in_critical_section:
                    # We want to request this resource
                    r = resource_map[needed_res_id]
                    # Enqueue in resource's global queue
                    item = {
                        "node": nd,
                        "arrival_time": time,
                        "on_critical_path": nd.on_critical_path,
                        "is_hard": nd.is_hard
                    }
                    r.enqueue_global(item, policy=policy)
                    # If we are at front, lock
                    front_item = r.front()
                    if front_item and front_item["node"] == nd:
                        nd.in_critical_section = True
                        nd.locked_resource = needed_res_id
                    else:
                        # blocked => we yield CPU
                        p.release()

                # If we are not blocked => run for 1 time unit
                if p.occupant == nd:  # still occupant
                    nd.remaining_time -= 1
                    nd.current_exec_progress += 1

                    if nd.remaining_time <= 0:
                        # finished node
                        if nd.in_critical_section and nd.locked_resource is not None:
                            # release from resource
                            rid = nd.locked_resource
                            r = resource_map[rid]
                            r.remove_item(nd)
                            nd.locked_resource = None
                            nd.in_critical_section = False
                        nd.completed = True
                        p.release()
                    else:
                        # if we are in critical section, we are non-preemptable
                        # do nothing more
                        pass

        # Deadline check
        for t in tasks:
            for nd in t.nodes:
                if nd.is_hard and not nd.completed:
                    if time >= nd.deadline:
                        schedulable = False
                        break
            if not schedulable:
                break
        if not schedulable:
            break

        time += 1

    return {
        "schedulable": schedulable,
        "finish_time": time
    }

# ================================================================
#                       MAIN
# ================================================================

def main():
    random.seed(0)

    # 1) Generate tasks
    tasks = []
    for i in range(NUM_TASKS):
        t = generate_one_task(i+1)
        tasks.append(t)

    # 2) Generate resources
    resources, resource_info = generate_resources(NUM_TASKS)

    # 3) Distribute partial usage intervals among tasks/nodes
    #    i.e., parse resource_info[r]["distribution"][task_idx] => that many intervals
    assign_resource_intervals_to_tasks(tasks, resources, resource_info)

    # Print federated info
    m_fed, m_simple = federated_output(tasks, resource_info)

    # 4) Schedule with FIFO
    fifo_result = schedule_with_pomip(tasks, resources, time_limit=1000, policy="FIFO")

    # We must reset node states to re-run
    for t in tasks:
        for nd in t.nodes:
            nd.reset_execution_state()

    # 5) Schedule with FPC
    fpc_result = schedule_with_pomip(tasks, resources, time_limit=1000, policy="FPC")

    print("===== POMIP (FIFO) =====")
    print("Schedulable?", fifo_result["schedulable"], "Finish time:", fifo_result["finish_time"])
    print("===== POMIP (FPC)  =====")
    print("Schedulable?", fpc_result["schedulable"], "Finish time:", fpc_result["finish_time"])


if __name__ == "__main__":
    main()

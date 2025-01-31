import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import deque

# --------------------------------------------------------
# 1) Global Parameter Settings (old variables)
# --------------------------------------------------------
NUM_TASKS = 5  # Number of tasks to generate
NODES_RANGE = (5, 20)      # Range for # of intermediate nodes
WCET_RANGE = (13, 30)      # Range for each node's WCET
P_EDGE = 0.1               # Probability of edge creation in DAG
D_RATIO_RANGE = (0.125, 0.25)
RESOURCE_RANGE = (1, 6)    # Range for # of shared resources
ACCESS_COUNT_RANGE = (1, 16)
ACCESS_LEN_RANGE = (5, 100)
CRITICAL_RATIOS = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
OUTPUT_DIR = "graphs_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# 2) Node, Resource, and Task Classes
# --------------------------------------------------------

class Node:
    def __init__(self, node_id, wcet, is_hard=False):
        self.node_id = node_id
        self.wcet = wcet
        self.is_hard = is_hard
        self.remaining_time = wcet
        self.exec_progress = 0  # how many time units of execution have happened
        self.resource_intervals = []  # list of (start, end, resource_id)
        self.in_critical_section = False
        self.completed = False
        self.locked_resource = None
        self.deadline = None  # optional, if you do per-node deadlines

    def add_resource_interval(self, start, end, resource_id):
        """
        e.g. from time offsets [start..end) in this node's total execution,
        it needs resource_id. Must have 0 <= start < end <= wcet
        """
        self.resource_intervals.append((start, end, resource_id))

    def get_resource_needed_now(self):
        """
        Check if current exec_progress is in any [start..end) interval.
        Return resource_id or None if no resource needed.
        """
        for (s, e, r_id) in self.resource_intervals:
            if s <= self.exec_progress < e:
                return r_id
        return None

    def reset(self):
        self.remaining_time = self.wcet
        self.exec_progress = 0
        self.in_critical_section = False
        self.completed = False
        self.locked_resource = None


class Resource:
    """
    Each resource l_q has a global queue GQ_q.
    We store items of the form:
      { "node": Node, "arrival_time": int, "is_hard": bool }
    If node is at front of GQ, it locks the resource => node.in_critical_section = True
    """
    def __init__(self, resource_id, max_access_length):
        self.resource_id = resource_id
        self.global_queue = deque()
        self.max_access_length = max_access_length

    def enqueue_global(self, item, policy="FIFO"):
        self.global_queue.append(item)
        if policy == "FPC":
            # "First-path-critical" or "Critical-first" approach
            # We'll do a simple approach: Hard nodes come first, then arrival_time.
            def priority(elem):
                return (not elem["is_hard"], elem["arrival_time"])
            sorted_list = sorted(self.global_queue, key=priority)
            self.global_queue = deque(sorted_list)

    def front(self):
        return self.global_queue[0] if self.global_queue else None

    def remove_item(self, node):
        for i, it in enumerate(self.global_queue):
            if it["node"] == node:
                del self.global_queue[i]
                break

    def __repr__(self):
        return f"Resource(id={self.resource_id})"
    
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


class Task:
    """
    Represents a single DAG-based task: 
      - node_types
      - Nx DAG
      - Real-time parameters (C, L, D, T, U)
    """
    def __init__(self, task_id, G, node_list, source_id, sink_id):
        self.task_id = task_id
        self.G = G
        self.node_list = node_list
        self.source_id = source_id
        self.sink_id = sink_id

        # Summaries
        self.C = 0
        self.L = 0
        self.D = 0
        self.T = 0
        self.U = 0
        self.critical_ratio = 0  # from random generation

    def compute_params(self):
        """
        C = sum of intermediate nodes' WCET
        L = longest path
        D = derived from L / random ratio in D_RATIO_RANGE
        U = C / D
        """
        # sum of wcets (excluding source/sink if 0)
        self.C = sum(n.wcet for n in self.node_list if n.wcet > 0)
        # compute L using node wcet
        wcet_map = {n.node_id: n.wcet for n in self.node_list}
        self.L = compute_longest_path_length_dag(self.G, wcet_map)

        ratio_d = random.uniform(*D_RATIO_RANGE)
        self.D = int(self.L / ratio_d) if ratio_d != 0 else self.L
        self.T = self.D
        self.U = self.C / self.T if self.T > 0 else float('inf')

    def reset_nodes(self):
        for nd in self.node_list:
            nd.reset()

    def __repr__(self):
        return (f"Task(id={self.task_id},C={self.C},L={self.L},"
                f"D={self.D},U={self.U:.2f})")


# --------------------------------------------------------
# 3) DAG Generation (original style)
# --------------------------------------------------------
def create_random_dag(num_inner, p):
    """
    Creates a random DAG with 'num_inner' intermediate nodes (0..num_inner-1),
    plus a source (id=num_inner) and sink (id=num_inner+1).
    Also randomly assigns some fraction as Hard per CRITICAL_RATIOS.
    Returns (G, node_types, ratio_crit).
    """
    total_nodes = num_inner + 2
    source_id = num_inner
    sink_id = num_inner + 1

    while True:
        G = nx.DiGraph()
        G.add_nodes_from(range(total_nodes))

        # 1) Node types
        node_types = {n: None for n in range(total_nodes)}
        for n in range(num_inner):
            node_types[n] = "Soft"

        # 2) random ratio from CRITICAL_RATIOS
        ratio_crit = random.choice(CRITICAL_RATIOS)
        ratio_crit /= (ratio_crit + 1)
        num_crit = round(ratio_crit * num_inner)
        crit_nodes = set(random.sample(range(num_inner), num_crit)) if num_inner > 0 else set()
        for c in crit_nodes:
            node_types[c] = "Hard"

        # 3) Add edges
        for u in range(num_inner):
            for v in range(u+1, num_inner):
                # avoid Soft->Hard constraint from your original code if desired
                if random.random() < p and not (node_types[u] == "Soft" and node_types[v] == "Hard"):
                    G.add_edge(u, v)

        # 4) ensure minimal connectivity
        if num_inner > 0:
            G.add_edge(source_id, 0)
            G.add_edge(num_inner-1, sink_id)
            for n in range(num_inner):
                if not nx.has_path(G, source_id, n):
                    G.add_edge(source_id, n)

        # 5) check acyclicity
        try:
            nx.topological_sort(G)
            return G, node_types, ratio_crit
        except nx.NetworkXUnfeasible:
            continue

def compute_longest_path_length_dag(G, wcet_map):
    """
    DP approach: for each node, dist[node] = wcet[node] + max of dist[predecessors].
    We'll do topological sort, then compute dist forward.
    """
    topo = list(nx.topological_sort(G))
    dist = {}
    for v in G.nodes():
        dist[v] = wcet_map[v]
    for u in topo:
        for v in G.successors(u):
            if dist[v] < dist[u] + wcet_map[v]:
                dist[v] = dist[u] + wcet_map[v]
    return max(dist.values()) if dist else 0

def generate_one_task(task_id):
    num_inner = random.randint(*NODES_RANGE)
    G_mid, node_types, ratio_crit = create_random_dag(num_inner, P_EDGE)
    source_id = num_inner
    sink_id = num_inner + 1
    total_nodes = num_inner + 2

    # Build final DAG
    G = nx.DiGraph()
    G.add_nodes_from(range(total_nodes))
    G.add_edges_from(G_mid.edges())
    if num_inner > 0:
        G.add_edge(source_id, 0)
        G.add_edge(num_inner-1, sink_id)

    # Build Node objects
    node_list = []
    for n in range(total_nodes):
        if n in (source_id, sink_id):
            w = 0
        else:
            w = random.randint(*WCET_RANGE)
        is_hard = (node_types[n] == "Hard")
        nd = Node(n, w, is_hard)
        nd.task_id = task_id
        node_list.append(nd)

    # Build Task
    t = Task(task_id, G, node_list, source_id, sink_id)
    t.critical_ratio = ratio_crit
    t.compute_params()
    for nd in t.node_list:
        nd.deadline = t.D

    # Optional: Plot and save
    plt.figure(figsize=(6, 5))
    pos = hierarchy_pos(G, source_id)
    node_colors = []
    for nd in node_list:
        if nd.node_id in (source_id, sink_id):
            node_colors.append("yellow")
        elif nd.is_hard:
            node_colors.append("red")
        else:
            node_colors.append("green")
    nx.draw_networkx_nodes(G, pos,
                           nodelist=[nd.node_id for nd in node_list],
                           node_color=node_colors,
                           node_size=500)
    nx.draw_networkx_edges(G, pos, arrows=True)
    labels_dict = {nd.node_id: str(nd.node_id) for nd in node_list}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_color='white')
    plt.title(f"Task {task_id} | N={num_inner} | ratio_crit={ratio_crit:.2f}")
    plt.axis('off')
    png_path = os.path.join(OUTPUT_DIR, f"dag_task_{task_id}.png")
    plt.savefig(png_path)
    plt.close()

    return t

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """ 
    Create a hierarchical layout for a tree-like DAG.
    """
    pos = {root: (xcenter, vert_loc)}
    neighbors = list(G.successors(root))
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width/2 - dx/2
        for neighbor in neighbors:
            nextx += dx
            pos.update(hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx))
    return pos

# --------------------------------------------------------
# 4) Resource Generation (like old code)
# --------------------------------------------------------
def generate_resources(n_r, num_tasks):
    """
    Return a list of Resource plus metadata for printing.
    """
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

def assign_resource_intervals_to_nodes(tasks, resources, resources_info):
    """
    Distribute resource usage intervals among the tasks' nodes, based on the
    'distribution' field in resources_info.
    
    :param tasks: list of Task objects. tasks[i] is the i-th task.
    :param resources: list of Resource objects, each with resource_id.
    :param resources_info: list of dict, each like:
       {
         "resource_id": <int>,
         "max_access_length": <int>,
         "total_accesses": <int>,
         "distribution": [acc0, acc1, ..., accN],  # how many times each task uses this resource
       }
    """
    # For each resource info, parse how many times each task needs that resource
    for r_info in resources_info:
        rid = r_info["resource_id"]
        max_len = r_info["max_access_length"]
        distribution = r_info["distribution"]  # e.g. [acc_t1, acc_t2, ...]

        # For each task index i, we have distribution[i] usage_count
        for i, usage_count in enumerate(distribution):
            if usage_count <= 0:
                continue

            # We'll distribute usage_count intervals among the nodes of tasks[i].
            # We'll pick random nodes that have wcet > 0. (Ignoring source/sink if wcet=0.)
            candidate_nodes = [nd for nd in tasks[i].node_list if nd.wcet > 0]

            for _ in range(usage_count):
                if not candidate_nodes:
                    # If, for some reason, we run out of nodes, break early
                    break

                # Pick a random node from this task
                nd = random.choice(candidate_nodes)

                # Decide an interval length up to min(max_len, nd.wcet).
                interval_length = random.randint(1, min(max_len, nd.wcet))
                # Pick a random start in [0..nd.wcet - interval_length]
                start = random.randint(0, nd.wcet - interval_length)
                end = start + interval_length

                # Add the interval to this node
                nd.add_resource_interval(start, end, rid)

                # Optional: If you don't want multiple intervals for the same node,
                # you could remove it from candidate_nodes. 
                # But typically multiple intervals are allowed. 
                # candidate_nodes.remove(nd)  # uncomment if each usage must map to a distinct node



# --------------------------------------------------------
# 5) Federated Scheduling Code
# --------------------------------------------------------

def compute_processors_federated(tasks):
    """
    If U_i > 1 => m_i = ceil((C_i - L_i)/(D_i - L_i)), else 1 CPU
    """
    total = 0
    for t in tasks:
        Ci = t.C
        Li = t.L
        Di = t.D
        Ui = t.U
        if Ui > 1:
            denom = (Di - Li) if (Di - Li) > 0 else 1
            top = (Ci - Li)
            m_i = math.ceil(top / denom)
            if m_i < 1:
                m_i = 1
            total += m_i
        else:
            total += 1
    return total

def federated_output(tasks, n_r):
    U_sum = sum(t.U for t in tasks)
    U_norm = random.uniform(0.1, 1)
    m_simple = math.ceil(U_sum / U_norm)
    m_fed = compute_processors_federated(tasks)

    print("==================================================")
    print(" Task Generation (Phase One) ")
    print("==================================================")
    for t in tasks:
        print(f"Task {t.task_id}: #nodes={len(t.node_list)}, ratio_crit={t.critical_ratio:.2f}")
        print(f"  C={t.C}, L={t.L}, D={t.D}, U={t.U:.3f}")

    print("----------------------------------")
    print(f"Total Utilization UΣ = {U_sum:.3f}")
    print("----------------------------------")

    print(f"\n#Resources (n_r) = {n_r}")

    print("\n==================================================")
    print(" Required Number of Processors ")
    print("==================================================")
    print(f"Based on total utilization => m_simple={m_simple}")
    print(f"Federated scheduling => m_fed={m_fed}")
    print("==================================================")

    return m_fed, m_simple

# --------------------------------------------------------
# 6) POMIP Scheduler with Local & Global Queues
# --------------------------------------------------------

class POMIPScheduler:
    """
    - We do a time-stepped approach with 1 CPU per task (federated).
    - local_queues[(task_id, resource_id)] for each resource & task.
    - Resource has global_queue GQ.
    - If node is in critical section but no CPU free => 
        * preempt occupant in normal section (Rule #1), 
        * else attempt migration (Rule #2).
    """
    def __init__(self, tasks, resources, policy="FIFO", time_limit=2000):
        self.tasks = tasks
        self.resources = resources
        self.policy = policy
        self.time_limit = time_limit

        # local_queues[(task_id, resource_id)] = deque(Node)
        self.local_queues = {}
        for t in tasks:
            for r in resources:
                self.local_queues[(t.task_id, r.resource_id)] = deque()

        # cluster_state[task_id] = [ occupant_node or None ] => 1 CPU
        self.cluster_state = {}
        for t in tasks:
            self.cluster_state[t.task_id] = [None]  # 1 CPU

        # resource_map
        self.resource_map = {r.resource_id: r for r in resources}

    def run(self):
        schedulable = True
        time = 0

        while time < self.time_limit:
            # check if all tasks done
            if self.all_done():
                schedulable = True
                break

            # for each cluster
            for t in self.tasks:
                cpus = self.cluster_state[t.task_id]
                for cpu_idx, occupant in enumerate(cpus):
                    if occupant is None:
                        # pick a node from the task
                        nd = self.pick_node_to_run(t)
                        if nd:
                            cpus[cpu_idx] = nd
                    else:
                        # occupant is running
                        nd = occupant
                        # 1) Check if the node needs a resource right now
                        resource_needed = nd.get_resource_needed_now()  
                        if resource_needed is not None and not nd.in_critical_section:
                            # we attempt to lock the resource (via local queue -> global queue)
                            res_obj = self.resource_map[resource_needed]
                            self.request_resource_local_queue(nd, res_obj, time)

                            # If STILL not in critical section => blocked => yield CPU
                            if not nd.in_critical_section:
                                cpus[cpu_idx] = None
                                # attempt migration if we can't preempt
                                # e.g.:
                                ok = self.attempt_migration_if_blocked(nd, resource_needed)
                                continue

                        # 2) If node is not blocked => "run" occupant for 1 time unit
                        nd.remaining_time -= 1
                        nd.exec_progress += 1

                        if nd.remaining_time <= 0:
                            # finished
                            if nd.in_critical_section and nd.locked_resource:
                                r_obj = self.resource_map[nd.locked_resource]
                                self.release_resource_local_queue(nd, r_obj)
                            nd.completed = True
                            cpus[cpu_idx] = None


            for t in self.tasks:
                for nd in t.node_list:
                    if nd.is_hard and not nd.completed:
                        # If the current time is >= node's deadline, the node missed its deadline
                        if time >= nd.deadline:
                            schedulable = False
                            break
                if not schedulable:
                    break

            if not schedulable:
                break
            time += 1
        if schedulable and not self.all_done():
            schedulable = False
        return {"schedulable": schedulable, "finish_time": time}

    def all_done(self):
        for t in self.tasks:
            for nd in t.node_list:
                if not nd.completed and nd.wcet > 0:
                    return False
        return True

    def pick_node_to_run(self, task):
        """
        Very simple approach: pick any incomplete node. 
        Priority to Hard if not in critical section.
        (You can add DAG precedence checks here.)
        """
        cands = []
        for nd in task.node_list:
            if not nd.completed and nd.remaining_time > 0:
                cands.append(nd)
        # Hard first
        c_hard = [x for x in cands if x.is_hard and not x.in_critical_section]
        c_soft = [x for x in cands if not x.is_hard]
        if c_hard:
            return c_hard[0]
        elif c_soft:
            return c_soft[0]
        else:
            return None

    # ----------- Local & Global Queue Logic -----------

    def request_resource_local_queue(self, node, resource_obj, current_time):
        # local queue
        tid = node.task_id
        rid = resource_obj.resource_id
        fq = self.local_queues[(tid, rid)]

        if node not in fq:
            fq.append(node)

        # if node is at front => try global
        if fq and fq[0] == node:
            item = {
                "node": node,
                "arrival_time": current_time,
                "is_hard": node.is_hard
            }
            resource_obj.enqueue_global(item, policy=self.policy)
            # if node is front in GQ => lock
            front_gq = resource_obj.front()
            if front_gq and front_gq["node"] == node:
                node.in_critical_section = True
                node.locked_resource = rid

    def release_resource_local_queue(self, node, resource_obj):
        # remove from GQ
        resource_obj.remove_item(node)
        # remove from local queue
        tid = node.task_id
        rid = resource_obj.resource_id
        fq = self.local_queues[(tid, rid)]
        if fq and fq[0] == node:
            fq.popleft()
        else:
            if node in fq:
                fq.remove(node)
        # clear
        node.in_critical_section = False
        node.locked_resource = None
        # next head in FQ tries to join GQ
        if fq:
            head_node = fq[0]
            item = {
                "node": head_node,
                "arrival_time": 0,
                "is_hard": head_node.is_hard
            }
            resource_obj.enqueue_global(item, policy=self.policy)
            front_gq = resource_obj.front()
            if front_gq and front_gq["node"] == head_node:
                head_node.in_critical_section = True
                head_node.locked_resource = rid

    # ----------- POMIP Rules #1, #2 -----------

    def attempt_preemption_in_cluster(self, cluster_cpus, node):
        """
        If no free CPU, node can preempt occupant if occupant is not in critical section
        (assuming node is in critical section).
        Return index of CPU if success, else None.
        """
        # 1) find free CPU
        for idx, occupant in enumerate(cluster_cpus):
            if occupant is None:
                return idx
        # 2) no free CPU => preempt occupant if occupant.in_critical_section = False
        if node.in_critical_section:
            for idx, occupant in enumerate(cluster_cpus):
                if occupant and not occupant.in_critical_section:
                    return idx
        return None

    def attempt_migration_if_blocked(self, node, resource_id):
        """
        If node can't run on home cluster => 
        check other clusters that also are blocked on resource_id => local_queues not empty.
        Then do a preemption attempt there.
        """
        home_tid = node.task_id
        for t in self.tasks:
            if t.task_id == home_tid:
                continue
            # check local_queues
            if len(self.local_queues[(t.task_id, resource_id)]) > 0:
                # that means they are also blocked
                cluster_cpus = self.cluster_state[t.task_id]
                idx = self.attempt_preemption_in_cluster(cluster_cpus, node)
                if idx is not None:
                    # do the migration
                    if cluster_cpus[idx] is not None:
                        cluster_cpus[idx] = None
                    cluster_cpus[idx] = node
                    return True
        return False

# --------------------------------------------------------
# 7) Main
# --------------------------------------------------------

def main():
    random.seed(1)

    # 1) Generate tasks
    tasks = []
    for i in range(NUM_TASKS):
        t_info = generate_one_task(i+1)
        tasks.append(t_info)

    # 2) Generate some resources
    n_r = random.randint(*RESOURCE_RANGE)
    resources, resources_info = generate_resources(n_r, NUM_TASKS)
    assign_resource_intervals_to_nodes(tasks, resources, resources_info)
    # 3) Print federated info
    m_fed, m_simple = federated_output(tasks, n_r)

    # 4) Run POMIP with FIFO
    #    We reset node states if needed
    for t in tasks:
        t.reset_nodes()
    sched_fifo = POMIPScheduler(tasks, resources, policy="FIFO", time_limit=2000)
    fifo_res = sched_fifo.run()
    print("===== POMIP (FIFO) =====")
    print("Schedulable?", fifo_res["schedulable"], "Finish time:", fifo_res["finish_time"])

    # 5) Run POMIP with FPC
    for t in tasks:
        t.reset_nodes()
    sched_fpc = POMIPScheduler(tasks, resources, policy="FPC", time_limit=2000)
    fpc_res = sched_fpc.run()
    print("===== POMIP (FPC) =====")
    print("Schedulable?", fpc_res["schedulable"], "Finish time:", fpc_res["finish_time"])


if __name__ == "__main__":
    main()

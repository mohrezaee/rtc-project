import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import deque

# --------------------------------------------------------
# 1) Global Parameter Settings (old variables)
# --------------------------------------------------------
NUM_TASKS = 10  # Number of tasks to generate
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
# NEW: Function to compute is_on_critical_path for each node
# --------------------------------------------------------
def compute_is_on_critical_path(G, node_types, source_id, sink_id):
    """
    A node v is on a 'critical path' if it lies on a path s->...->t that 
    includes at least one Hard node (which may be v or another node).
    Steps:
      1. Identify set ST of nodes that lie on *some* path from s to t.
      2. Among ST, find which nodes are Hard => call that set H'.
      3. For each h in H', do BFS forward/backward in ST => any node 
         reachable from or reaching h is on a path with h => mark them.
    Return a dict is_crit[v] = True/False.
    """
    # 1) Find all reachable from s
    reachable_from_s = set()
    queue = [source_id]
    while queue:
        u = queue.pop()
        for v in G.successors(u):
            if v not in reachable_from_s:
                reachable_from_s.add(v)
                queue.append(v)
    # 2) Find all that can reach t (reverse BFS)
    G_rev = G.reverse()
    can_reach_t = set()
    queue = [sink_id]
    while queue:
        u = queue.pop()
        for v in G_rev.successors(u):
            if v not in can_reach_t:
                can_reach_t.add(v)
                queue.append(v)

    ST = set()
    for v in G.nodes():
        if v in reachable_from_s and v in can_reach_t:
            ST.add(v)

    # Among ST, find Hard nodes
    hard_nodes = [v for v in ST if node_types[v] == "Hard"]

    if not hard_nodes:
        return {v: False for v in G.nodes()}

    # BFS forward in subgraph ST, BFS backward in subgraph ST
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

    critical_set = set()
    for h in hard_nodes:
        fwd = bfs_forward(h)
        bwd = bfs_backward(h)
        critical_set.update(fwd)
        critical_set.update(bwd)

    is_crit = {}
    for v in G.nodes():
        is_crit[v] = (v in ST) and (v in critical_set)
    return is_crit

# --------------------------------------------------------
# 2) Node, Resource, and Task Classes
# --------------------------------------------------------

class Node:
    def __init__(self, node_id, wcet, is_hard=False):
        self.node_id = node_id
        self.wcet = wcet
        self.is_hard = is_hard
        self.remaining_time = wcet
        self.exec_progress = 0
        self.resource_intervals = []
        self.in_critical_section = False
        self.completed = False
        self.locked_resource = None
        self.deadline = None
        self.task_id = None
        self.finish_time = None
        self.home_task_id = None

        # NEW: We'll track this after we compute it
        self.is_on_critical_path = False

    def add_resource_interval(self, start, end, resource_id):
        self.resource_intervals.append((start, end, resource_id))

    def get_resource_needed_now(self):
        for (s, e, r_id) in self.resource_intervals:
            if s <= self.exec_progress < e:
                return r_id
        return None
    
    def should_release_locked_resource(self):
        if self.locked_resource is None:
            return False
        for (s, e, r_id) in self.resource_intervals:
            if r_id == self.locked_resource and (s <= self.exec_progress < e):
                return False
        return True

    def reset(self):
        self.remaining_time = self.wcet
        self.exec_progress = 0
        self.in_critical_section = False
        self.completed = False
        self.locked_resource = None
        self.finish_time = None
        # is_on_critical_path remains the same across runs (or reset if you prefer)

class Resource:
    def __init__(self, resource_id, max_access_length):
        self.resource_id = resource_id
        self.global_queue = deque()

    def enqueue_global(self, item, policy="FIFO"):
        """
        item = {
          "node": node_obj,
          "arrival_time": current_time,
          "is_critical": node_obj.is_hard,
          "is_crit_path": node_obj.is_on_critical_path
        }
        """
        self.global_queue.append(item)
        if policy == "FPC":
            # Sort by "on_critical_path" = True first, tie-break arrival_time
            def priority(elem):
                # 'is_crit_path' descending => we do (not elem["is_crit_path"]) 
                # then arrival_time ascending
                return (not elem["is_crit_path"], elem["arrival_time"])
            sorted_list = sorted(self.global_queue, key=priority)
            self.global_queue = deque(sorted_list)

    def front(self):
        return self.global_queue[0] if self.global_queue else None

    def remove_item(self, node):
        for i, item in enumerate(self.global_queue):
            if item['node'] == node:
                del self.global_queue[i]
                break

class Task:
    def __init__(self, task_id, G, node_list, source_id, sink_id):
        self.task_id = task_id
        self.G = G
        self.node_list = node_list
        self.source_id = source_id
        self.sink_id = sink_id
        self.C = sum(n.wcet for n in node_list if n.wcet > 0)
        self.L = compute_longest_path_length_dag(G, {n.node_id: n.wcet for n in node_list})
        ratio_d = random.uniform(0.125, 0.25)
        self.D = int(self.L / ratio_d) if ratio_d > 0 else self.L
        self.T = self.D
        self.U = self.C / self.T if self.T > 0 else float('inf')
        for n in node_list:
            n.deadline = self.D

    def reset_nodes(self):
        for node in self.node_list:
            node.reset()


# --------------------------------------------------------
# 3) DAG Generation
# --------------------------------------------------------
def create_random_dag(num_inner, p):
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
        for c in crit_nodes:
            node_types[c] = "Hard"

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

def compute_longest_path_length_dag(G, wcet_map):
    topo_order = list(nx.topological_sort(G))
    dist = {v: wcet_map[v] for v in G.nodes}
    for u in topo_order:
        for v in G.successors(u):
            dist[v] = max(dist[v], dist[u] + wcet_map[v])
    return max(dist.values())

def generate_one_task(task_id):
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

    node_list = []
    for n in range(total_nodes):
        if n in (source_id, sink_id):
            w = 0
        else:
            w = random.randint(*WCET_RANGE)
        is_hard = (node_types[n] == "Hard")
        nd = Node(n, w, is_hard)
        nd.task_id = task_id
        nd.home_task_id = task_id
        node_list.append(nd)

    t = Task(task_id, G, node_list, source_id, sink_id)
    t.critical_ratio = ratio_crit

    # NEW: Compute is_on_critical_path for each node
    is_crit_map = compute_is_on_critical_path(G, node_types, source_id, sink_id)
    for nd in t.node_list:
        nd.is_on_critical_path = is_crit_map[nd.node_id]
        # (We also assigned nd.deadline in Task constructor)

    return t

# --------------------------------------------------------
# 4) Resource Generation
# --------------------------------------------------------
def generate_resources(n_r, num_tasks):
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
    for r_info in resources_info:
        rid = r_info["resource_id"]
        max_len = r_info["max_access_length"]
        distribution = r_info["distribution"]

        for i, usage_count in enumerate(distribution):
            if usage_count <= 0:
                continue
            candidate_nodes = [nd for nd in tasks[i].node_list if nd.wcet > 0]

            while usage_count > 0 and candidate_nodes:
                node = random.choice(candidate_nodes)
                candidate_nodes.remove(node)

                for _ in range(usage_count):
                    interval_length = random.randint(1, min(max_len, node.wcet))
                    for _ in range(10):
                        start = random.randint(0, node.wcet - interval_length)
                        end = start + interval_length
                        conflict = any(
                            (s < end and start < e) and (r == rid)
                            for (s, e, r) in node.resource_intervals
                        )
                        if not conflict:
                            node.add_resource_interval(start, end, rid)
                            usage_count -= 1
                            break
                    if usage_count <= 0:
                        break

# --------------------------------------------------------
# 5) Federated Scheduling Code
# --------------------------------------------------------

def compute_processors_federated(tasks):
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
# 6) POMIP Scheduler with Local & Global Queues + Logging
# --------------------------------------------------------

class POMIPScheduler:
    def __init__(self, tasks, resources, policy="FIFO", time_limit=200):
        self.tasks = tasks
        self.resources = resources
        self.policy = policy
        self.time_limit = time_limit
        self.local_queues = {(t.task_id, r.resource_id): deque() for t in tasks for r in resources}
        self.cluster_state = {t.task_id: [None] for t in tasks}
        self.resource_map = {r.resource_id: r for r in resources}
        self.schedule_log = []

    def run(self):
        time = 0
        schedulable = True
        while time < self.time_limit:
            if self.all_done():
                break

            time_snapshot = {}

            for task in self.tasks:
                occupant = self.cluster_state[task.task_id][0]
                if occupant is None:
                    node = self.pick_node_to_run(task)
                    if node:
                        self.cluster_state[task.task_id][0] = node
                        occupant = node

                if occupant:
                    resource_id = occupant.get_resource_needed_now()
                    if resource_id and not occupant.in_critical_section:
                        res = self.resource_map[resource_id]
                        self.request_resource_local_queue(occupant, res, time)

                        if not occupant.in_critical_section:
                            preempted = self.apply_rule_one(task, occupant)
                            if not preempted:
                                migrated = self.apply_rule_two(occupant, resource_id)
                                if not migrated:
                                    self.cluster_state[task.task_id][0] = None
                            occupant = self.cluster_state[task.task_id][0]

                    if occupant:
                        occupant.remaining_time -= 1
                        occupant.exec_progress += 1

                        if occupant.locked_resource and occupant.should_release_locked_resource():
                            self.release_resource_local_queue(occupant, self.resource_map[occupant.locked_resource])
                            if occupant.task_id != occupant.home_task_id:
                                self.return_to_original_task(occupant)

                        if occupant.remaining_time <= 0:
                            occupant.completed = True
                            occupant.finish_time = time + 1
                            self.cluster_state[task.task_id][0] = None

                # Log occupant
                occupant_final = self.cluster_state[task.task_id][0]
                if occupant_final is not None:
                    time_snapshot[task.task_id] = (occupant_final.node_id, occupant_final.locked_resource)
                else:
                    time_snapshot[task.task_id] = (None, None)

            # Deadline check
            for t in self.tasks:
                for node in t.node_list:
                    if node.is_hard and not node.completed and time >= node.deadline:
                        schedulable = False
                        break
                if not schedulable:
                    break
            if not schedulable:
                break

            self.schedule_log.append(time_snapshot)
            time += 1

        avg_qos = self.compute_qos()
        return {"schedulable": schedulable, "finish_time": time, "average_qos": avg_qos}

    def apply_rule_one(self, task, blocking_node):
        for idx, occupant in enumerate(self.cluster_state[task.task_id]):
            if occupant and not occupant.in_critical_section:
                self.cluster_state[task.task_id][idx] = blocking_node
                return True
        return False

    def apply_rule_two(self, blocking_node, resource_id):
        for t in self.tasks:
            if t.task_id == blocking_node.task_id:
                continue
            for idx, occupant in enumerate(self.cluster_state[t.task_id]):
                if occupant is None:
                    if blocking_node.home_task_id is None:
                        blocking_node.home_task_id = blocking_node.task_id
                    self.cluster_state[t.task_id][idx] = blocking_node
                    blocking_node.task_id = t.task_id
                    return True
                else:
                    if not occupant.in_critical_section:
                        if blocking_node.home_task_id is None:
                            blocking_node.home_task_id = blocking_node.task_id
                        self.cluster_state[t.task_id][idx] = blocking_node
                        blocking_node.task_id = t.task_id
                        return True
        return False

    def return_to_original_task(self, node):
        orig_id = node.home_task_id
        for idx, occupant in enumerate(self.cluster_state[orig_id]):
            if occupant is None:
                self.cluster_state[orig_id][idx] = node
                node.task_id = orig_id
                node.home_task_id = None
                return

    def compute_qos(self):
        total_qos = 0.0
        count = 0
        for task in self.tasks:
            for node in task.node_list:
                if not node.is_hard and node.wcet>0:
                    count += 1
                    if node.finish_time is None:
                        total_qos += 0
                    else:
                        delay = max(0, node.finish_time - node.deadline)
                        q = max(0, 1.0 - 0.30*delay)
                        total_qos += q
        return total_qos / count if count else 1.0
    
    def all_done(self):
        return all(n.completed for t in self.tasks for n in t.node_list if n.wcet>0)

    def pick_node_to_run(self, task):
        cands = [n for n in task.node_list if not n.completed]
        h = [n for n in cands if n.is_hard and not n.in_critical_section]
        if h:
            return h[0]
        s = [n for n in cands if not n.is_hard]
        if s:
            return s[0]
        return None

    def request_resource_local_queue(self, node, resource, time):
        key = (node.task_id, resource.resource_id)
        fq = self.local_queues[key]
        if node not in fq:
            fq.append(node)
        if fq and fq[0] == node:
            # Enqueue with extra field is_crit_path=node.is_on_critical_path
            item = {
                "node": node,
                "arrival_time": time,
                "is_critical": node.is_hard,
                "is_crit_path": node.is_on_critical_path
            }
            resource.enqueue_global(item, policy=self.policy)
            front = resource.front()
            if front and front["node"] == node:
                node.in_critical_section = True
                node.locked_resource = resource.resource_id
                return True
        return False

    def release_resource_local_queue(self, node, resource):
        resource.remove_item(node)
        key = (node.task_id, resource.resource_id)
        fq = self.local_queues[key]
        if fq and fq[0] == node:
            fq.popleft()
        node.in_critical_section = False
        node.locked_resource = None
        if fq:
            head_node = fq[0]
            item = {
                "node": head_node,
                "arrival_time": 0,
                "is_critical": head_node.is_hard,
                "is_crit_path": head_node.is_on_critical_path
            }
            resource.enqueue_global(item, policy=self.policy)
            front = resource.front()
            if front and front["node"] == head_node:
                head_node.in_critical_section = True
                head_node.locked_resource = resource.resource_id

    def plot_schedule(self, title="POMIP Schedule", filename="schedule.png"):
        fig, ax = plt.subplots(figsize=(10, 3 + len(self.tasks)*0.5))

        intervals_per_cluster = {t.task_id: [] for t in self.tasks}

        if not self.schedule_log:
            plt.title(title + " (No execution recorded)")
            plt.savefig(filename)
            plt.close()
            return

        max_time = len(self.schedule_log)
        for t_idx in range(max_time):
            snapshot = self.schedule_log[t_idx]
            for task_id, (nid, locked_r) in snapshot.items():
                intervals = intervals_per_cluster[task_id]
                if not intervals:
                    intervals.append([t_idx, t_idx+1, nid, locked_r])
                else:
                    last = intervals[-1]
                    if last[2] == nid and last[3] == locked_r:
                        last[1] = t_idx+1
                    else:
                        intervals.append([t_idx, t_idx+1, nid, locked_r])

        for i, t in enumerate(self.tasks):
            y = i
            intervals = intervals_per_cluster[t.task_id]
            for (start, end, occupant_id, locked_r) in intervals:
                if occupant_id is not None:
                    occupant_node = None
                    for nd in t.node_list:
                        if nd.node_id == occupant_id:
                            occupant_node = nd
                            break
                    if occupant_node and occupant_node.is_hard:
                        color = 'red'
                    else:
                        color = 'green'

                    length = end - start
                    ax.barh(y, length, left=start, height=0.4, color=color,
                            edgecolor='black', alpha=0.8)

                    label_str = str(occupant_id)
                    if locked_r is not None:
                        label_str += f"(R{locked_r})"
                    ax.text((start+end)/2, y, label_str, ha='center', va='center', 
                            color='white', fontsize=8)
                else:
                    pass

            ax.text(-2, y, f"Task {t.task_id}", va='center', fontsize=9)

        ax.set_xlim(0, max_time)
        ax.set_ylim(-1, len(self.tasks))
        ax.set_xlabel("Time")
        ax.set_ylabel("Task (Cluster)")
        ax.set_yticks([])
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()


# --------------------------------------------------------
# 7) Main
# --------------------------------------------------------

def main():
    random.seed(0)

    tasks = []
    for i in range(NUM_TASKS):
        t_info = generate_one_task(i+1)
        tasks.append(t_info)

    n_r = random.randint(*RESOURCE_RANGE)
    resources, resources_info = generate_resources(n_r, NUM_TASKS)
    assign_resource_intervals_to_nodes(tasks, resources, resources_info)

    m_fed, m_simple = federated_output(tasks, n_r)

    # FIFO
    for t in tasks:
        t.reset_nodes()
    sched_fifo = POMIPScheduler(tasks, resources, policy="FIFO", time_limit=200)
    fifo_result = sched_fifo.run()
    print("\n=== POMIP (FIFO) ===")
    print("Schedulable?", fifo_result["schedulable"])
    print("Finish time:", fifo_result["finish_time"])
    print("Avg QoS:", fifo_result["average_qos"])
    sched_fifo.plot_schedule(title="POMIP-FIFO Gantt", filename="fifo_schedule.png")

    # FPC (we interpret "FPC" as "sort by critical-path first")
    for t in tasks:
        t.reset_nodes()
    sched_fpc = POMIPScheduler(tasks, resources, policy="FPC", time_limit=200)
    fpc_result = sched_fpc.run()
    print("\n=== POMIP (FPC) ===")
    print("Schedulable?", fpc_result["schedulable"])
    print("Finish time:", fpc_result["finish_time"])
    print("Avg QoS:", fpc_result["average_qos"])
    sched_fpc.plot_schedule(title="POMIP-FPC Gantt", filename="fpc_schedule.png")


if __name__ == "__main__":
    main()

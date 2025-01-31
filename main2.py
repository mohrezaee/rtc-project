import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import deque

# --------------------------------------------------------
# 1) Global Parameter Settings (old variables)
# --------------------------------------------------------
NUM_TASKS = 2  # Number of tasks to generate
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
        self.exec_progress = 0
        self.resource_intervals = []  # list of (start, end, resource_id)
        self.in_critical_section = False
        self.completed = False
        self.locked_resource = None
        self.deadline = None  # assigned later
        self.task_id = None
        self.finish_time = None

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
            if s <= self.exec_progress < e and r_id == self.locked_resource:
                return False
        return True

    def reset(self):
        self.remaining_time = self.wcet
        self.exec_progress = 0
        self.in_critical_section = False
        self.completed = False
        self.locked_resource = None
        self.finish_time = None

class Resource:
    def __init__(self, resource_id, max_access_length):
        self.resource_id = resource_id
        self.global_queue = deque()

    def enqueue_global(self, item, policy="FIFO"):
        self.global_queue.append(item)
        if policy == "CPF":
            self.global_queue = deque(sorted(self.global_queue, key=lambda x: (not x['is_critical'], x['arrival_time'])))

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
        node_list.append(nd)

    t = Task(task_id, G, node_list, source_id, sink_id)
    t.critical_ratio = ratio_crit

    # Assign deadlines to each node = T.D
    for nd in t.node_list:
        nd.deadline = t.D

    # optional: plot
    return t

def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
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

                # Avoid overlapping intervals or nested requests by checking conflicts
                current_intervals = node.resource_intervals

                for _ in range(usage_count):
                    # Determine an interval length within the allowable bounds
                    interval_length = random.randint(1, min(max_len, node.wcet))

                    # Try multiple times to find a valid non-conflicting interval
                    for _ in range(10):  # Give 10 tries to find a valid interval
                        start = random.randint(0, node.wcet - interval_length)
                        end = start + interval_length

                        # Check for conflicts
                        conflict = any(
                            (s < end and start < e) and (r == rid)  # Overlap with same resource
                            for (s, e, r) in current_intervals
                        )

                        if not conflict:
                            node.add_resource_interval(start, end, rid)
                            usage_count -= 1
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
    def __init__(self, tasks, resources, policy="FIFO", time_limit=2000):
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
                cpu = self.cluster_state[task.task_id][0]
                if cpu is None:
                    node = self.pick_node_to_run(task)
                    if node:
                        self.cluster_state[task.task_id][0] = node

                occupant = self.cluster_state[task.task_id][0]
                if occupant:
                    resource_id = occupant.get_resource_needed_now()
                    if resource_id and not occupant.in_critical_section:
                        resource = self.resource_map[resource_id]
                        self.request_resource_local_queue(occupant, resource, time)
                        if not occupant.in_critical_section:
                            self.cluster_state[task.task_id][0] = None
                            continue

                    occupant.remaining_time -= 1
                    occupant.exec_progress += 1
                    if occupant.locked_resource and occupant.should_release_locked_resource():
                        self.release_resource_local_queue(occupant, self.resource_map[occupant.locked_resource])
                    if occupant.remaining_time <= 0:
                        occupant.completed = True
                        occupant.finish_time = time + 1
                        self.cluster_state[task.task_id][0] = None

                time_snapshot[task.task_id] = occupant.node_id if occupant else None

            for task in self.tasks:
                for node in task.node_list:
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

    def compute_qos(self):
        total_qos = 0.0
        count = 0
        for task in self.tasks:
            for node in task.node_list:
                if not node.is_hard and node.wcet > 0:
                    count += 1
                    if node.finish_time is None:
                        total_qos += 0
                    else:
                        delay = max(0, node.finish_time - node.deadline)
                        total_qos += max(0, 1.0 - 0.30 * delay)
        return total_qos / count if count else 1.0

    def all_done(self):
        return all(node.completed for task in self.tasks for node in task.node_list if node.wcet > 0)

    def pick_node_to_run(self, task):
        candidates = [n for n in task.node_list if not n.completed]
        hard_nodes = [n for n in candidates if n.is_hard]
        if hard_nodes:
            return hard_nodes[0]
        soft_nodes = [n for n in candidates if not n.is_hard]
        return soft_nodes[0] if soft_nodes else None

    def request_resource_local_queue(self, node, resource, time):
        fq = self.local_queues[(node.task_id, resource.resource_id)]
        if node not in fq:
            fq.append(node)
        if fq[0] == node:
            resource.enqueue_global({"node": node, "arrival_time": time, "is_critical": node.is_hard}, self.policy)
            if resource.front()['node'] == node:
                node.in_critical_section = True
                node.locked_resource = resource.resource_id

    def release_resource_local_queue(self, node, resource):
        resource.remove_item(node)
        fq = self.local_queues[(node.task_id, resource.resource_id)]
        if fq and fq[0] == node:
            fq.popleft()
        node.in_critical_section = False
        node.locked_resource = None

    # ----------- POMIP Rules #1, #2 -----------

    def attempt_preemption_in_cluster(self, cluster_cpus, node):
        for idx, occupant in enumerate(cluster_cpus):
            if occupant is None:
                return idx
        if node.in_critical_section:
            for idx, occupant in enumerate(cluster_cpus):
                if occupant and not occupant.in_critical_section:
                    return idx
        return None

    def attempt_migration_if_blocked(self, node, resource_id):
        home_tid = node.task_id
        for t in self.tasks:
            if t.task_id == home_tid:
                continue
            if len(self.local_queues[(t.task_id, resource_id)]) > 0:
                cluster_cpus = self.cluster_state[t.task_id]
                idx = self.attempt_preemption_in_cluster(cluster_cpus, node)
                if idx is not None:
                    if cluster_cpus[idx] is not None:
                        cluster_cpus[idx] = None
                    cluster_cpus[idx] = node
                    return True
        return False

    # ---------- Visualization for Gantt ----------

    def plot_schedule(self, title="POMIP Schedule", filename="schedule.png"):
        """
        Create a Gantt-like chart from self.schedule_log:
          self.schedule_log[time][task_id] = node_id or None
        Each cluster is a "row" in the chart, time is the x-axis.
        We color nodes by node_id or is_hard status.
        """
        # We'll create a figure with one row per task
        fig, ax = plt.subplots(figsize=(10, 3 + len(self.tasks)*0.5))

        # Build an array: schedule_log[t][tid] = node_id or None
        # We'll store intervals as (start, end, occupant) for each cluster
        intervals_per_cluster = {t.task_id: [] for t in self.tasks}

        if not self.schedule_log:
            plt.title(title + " (No execution recorded)")
            plt.savefig(filename)
            plt.close()
            return

        max_time = len(self.schedule_log)
        # We'll parse the log
        # For each cluster (task_id), we gather consecutive time intervals where occupant = same node_id
        for t_idx in range(max_time):
            snapshot = self.schedule_log[t_idx]
            for task_id, node_id in snapshot.items():
                intervals = intervals_per_cluster[task_id]
                if not intervals:
                    # new interval
                    intervals.append([t_idx, t_idx+1, node_id]) # [start, end, occupant]
                else:
                    last = intervals[-1]
                    if last[2] == node_id:
                        # extend
                        last[1] = t_idx+1
                    else:
                        # new
                        intervals.append([t_idx, t_idx+1, node_id])

        # Now we plot each cluster as a "row"
        # cluster y = (some index), occupant is a rectangle from [start..end]
        ybase = 0
        for i, t in enumerate(self.tasks):
            y = i  # row index
            intervals = intervals_per_cluster[t.task_id]
            for (start, end, occupant) in intervals:
                if occupant is not None:
                    # color for occupant
                    # let's say if occupant is Hard => color='red', else 'green'
                    node_obj = None
                    for nd in t.node_list:
                        if nd.node_id == occupant:
                            node_obj = nd
                            break
                    if node_obj and node_obj.is_hard:
                        color = 'red'
                    else:
                        color = 'green'
                    ax.barh(y, end-start, left=start, height=0.4, color=color,
                            edgecolor='black', alpha=0.8)

                    ax.text((start+end)/2, y, f"{occupant}", ha='center', va='center', color='white', fontsize=8)

            ax.text(-2, y, f"Task {t.task_id}", va='center', fontsize=9)

        ax.set_xlim(0, max_time)
        ax.set_ylim(-1, len(self.tasks))
        ax.set_xlabel("Time")
        ax.set_ylabel("Cluster( Task )")
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

    # 1) Generate tasks
    tasks = []
    for i in range(NUM_TASKS):
        t_info = generate_one_task(i+1)
        tasks.append(t_info)

    # 2) Generate resources
    n_r = random.randint(*RESOURCE_RANGE)
    resources, resources_info = generate_resources(n_r, NUM_TASKS)

    # 3) Assign partial intervals based on distribution
    assign_resource_intervals_to_nodes(tasks, resources, resources_info)

    # 4) Print fed info
    m_fed, m_simple = federated_output(tasks, n_r)

    # 5) POMIP with FIFO
    for t in tasks:
        t.reset_nodes()
    sched_fifo = POMIPScheduler(tasks, resources, policy="FIFO", time_limit=200)
    fifo_result = sched_fifo.run()
    print("\n=== POMIP (FIFO) ===")
    print("Schedulable?", fifo_result["schedulable"])
    print("Finish time:", fifo_result["finish_time"])
    print("Avg QoS:", fifo_result["average_qos"])
    # Visualize
    sched_fifo.plot_schedule(title="POMIP-FIFO Gantt", filename="fifo_schedule.png")

    # 6) POMIP with FPC
    for t in tasks:
        t.reset_nodes()
    sched_fpc = POMIPScheduler(tasks, resources, policy="FPC", time_limit=200)
    fpc_result = sched_fpc.run()
    print("\n=== POMIP (FPC) ===")
    print("Schedulable?", fpc_result["schedulable"])
    print("Finish time:", fpc_result["finish_time"])
    print("Avg QoS:", fpc_result["average_qos"])
    # Visualize
    sched_fpc.plot_schedule(title="POMIP-FPC Gantt", filename="fpc_schedule.png")

if __name__ == "__main__":
    main()

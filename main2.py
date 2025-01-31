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
        self.deadline = None  # assigned later
        self.task_id = None

        # For QoS: We'll track the finishing time
        self.finish_time = None
        # We might also track start_time, but you can add that if needed.

    def add_resource_interval(self, start, end, resource_id):
        self.resource_intervals.append((start, end, resource_id))

    def get_resource_needed_now(self):
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
        self.finish_time = None

class Resource:
    def __init__(self, resource_id, max_access_length):
        self.resource_id = resource_id
        self.global_queue = deque()
        self.max_access_length = max_access_length

    def enqueue_global(self, item, policy="FIFO"):
        self.global_queue.append(item)
        if policy == "FPC":
            # Hard nodes come first, tie-break arrival_time
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

class Task:
    def __init__(self, task_id, G, node_list, source_id, sink_id):
        self.task_id = task_id
        self.G = G
        self.node_list = node_list
        self.source_id = source_id
        self.sink_id = sink_id

        self.C = 0
        self.L = 0
        self.D = 0
        self.T = 0
        self.U = 0
        self.critical_ratio = 0

    def compute_params(self):
        self.C = sum(n.wcet for n in self.node_list if n.wcet > 0)
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
    t.compute_params()

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
            for _ in range(usage_count):
                if not candidate_nodes:
                    break
                nd = random.choice(candidate_nodes)
                interval_length = random.randint(1, min(max_len, nd.wcet))
                start = random.randint(0, nd.wcet - interval_length)
                end = start + interval_length
                nd.add_resource_interval(start, end, rid)

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
    """
    - 1 CPU per task (federated).
    - local_queues[(task_id, resource_id)] for each resource & task.
    - We log scheduling decisions to produce a Gantt chart later.
    - If any hard node misses its deadline => schedulable= False
    - For non-critical nodes finishing after their deadline => QoS penalty
    """
    def __init__(self, tasks, resources, policy="FIFO", time_limit=2000):
        self.tasks = tasks
        self.resources = resources
        self.policy = policy
        self.time_limit = time_limit

        self.local_queues = {}
        for t in tasks:
            for r in resources:
                self.local_queues[(t.task_id, r.resource_id)] = deque()

        self.cluster_state = {}
        for t in tasks:
            self.cluster_state[t.task_id] = [None]  # 1 CPU for each task

        self.resource_map = {r.resource_id: r for r in resources}

        # Logging: schedule_log[time][ (task_id) ] = node_id or None
        # We'll store a record for each time step, for each cluster
        self.schedule_log = []

    def run(self):
        schedulable = True
        time = 0
        while time < self.time_limit:
            if self.all_done():
                break

            # We'll record occupant for each cluster this time step
            time_snapshot = {}

            for t in self.tasks:
                cpus = self.cluster_state[t.task_id]
                occupant = cpus[0]
                if occupant is None:
                    nd = self.pick_node_to_run(t)
                    if nd:
                        cpus[0] = nd
                        occupant = nd
                if occupant is not None:
                    # occupant runs
                    resource_needed = occupant.get_resource_needed_now()
                    if resource_needed is not None and not occupant.in_critical_section:
                        # request resource
                        r_obj = self.resource_map[resource_needed]
                        self.request_resource_local_queue(occupant, r_obj, time)
                        if not occupant.in_critical_section:
                            # blocked => release CPU
                            cpus[0] = None
                            # attempt migration if can't run in home cluster
                            ok = self.attempt_migration_if_blocked(occupant, resource_needed)
                            occupant = None

                # If occupant is still occupant, run
                if occupant is not None:
                    occupant.remaining_time -= 1
                    occupant.exec_progress += 1
                    if occupant.remaining_time <= 0:
                        occupant.finish_time = time+1  # record finishing
                        if occupant.in_critical_section and occupant.locked_resource:
                            r_obj = self.resource_map[occupant.locked_resource]
                            self.release_resource_local_queue(occupant, r_obj)
                        occupant.completed = True
                        cpus[0] = None

                # Record occupant node_id or None
                if occupant is not None:
                    time_snapshot[t.task_id] = occupant.node_id
                else:
                    time_snapshot[t.task_id] = None

            # After the time-step, check deadlines
            for t in self.tasks:
                for nd in t.node_list:
                    if nd.is_hard and not nd.completed:
                        if time >= nd.deadline:
                            # Critical node missed deadline => entire tasks unschedulable
                            schedulable = False
                            break
                if not schedulable:
                    break
            if not schedulable:
                break

            self.schedule_log.append(time_snapshot)
            time += 1

        # If we exit the loop but still have undone nodes => partial
        if schedulable and not self.all_done():
            schedulable = False

        # Compute final QoS
        avg_qos = self.compute_qos()

        return {
            "schedulable": schedulable,
            "finish_time": time,
            "average_qos": avg_qos
        }

    def compute_qos(self):
        """
        For each non-critical node:
          - If it finishes <= node.deadline => QoS=100%
          - If it finishes after => subtract 30% for each unit after
          - min QoS is 0
        Then average across all non-critical nodes
        """
        total_qos = 0.0
        count_noncrit = 0
        for t in self.tasks:
            for nd in t.node_list:
                if nd.wcet<=0:
                    continue
                if nd.is_hard:
                    # we only compute QoS for non-critical
                    continue
                count_noncrit += 1
                if nd.finish_time is None:
                    # never finished => QoS=0
                    total_qos += 0
                    continue
                # check deadline
                overrun = nd.finish_time - nd.deadline
                if overrun <= 0:
                    # finished on time
                    total_qos += 1.0
                else:
                    # each time unit => -30%
                    penalty = 0.30
                    qval = 1.0 - penalty
                    if qval < 0:
                        qval = 0
                    total_qos += qval
        if count_noncrit == 0:
            return 1.0  # or 0, or 1 meaning no non-critical => no QoS penalty
        return total_qos / count_noncrit

    def all_done(self):
        for t in self.tasks:
            for nd in t.node_list:
                if not nd.completed and nd.wcet>0:
                    return False
        return True

    def pick_node_to_run(self, task):
        cands = []
        for nd in task.node_list:
            if not nd.completed and nd.remaining_time>0:
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
        tid = node.task_id
        rid = resource_obj.resource_id
        fq = self.local_queues[(tid, rid)]

        if node not in fq:
            fq.append(node)

        if fq and fq[0] == node:
            item = {
                "node": node,
                "arrival_time": current_time,
                "is_hard": node.is_hard
            }
            resource_obj.enqueue_global(item, policy=self.policy)
            front_gq = resource_obj.front()
            if front_gq and front_gq["node"] == node:
                node.in_critical_section = True
                node.locked_resource = rid

    def release_resource_local_queue(self, node, resource_obj):
        resource_obj.remove_item(node)
        tid = node.task_id
        rid = resource_obj.resource_id
        fq = self.local_queues[(tid, rid)]
        if fq and fq[0] == node:
            fq.popleft()
        else:
            if node in fq:
                fq.remove(node)
        node.in_critical_section = False
        node.locked_resource = None
        # next head
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
    random.seed(1)

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

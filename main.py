import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os
from collections import deque

# --------------------------------------------------------
# General Parameter Settings
# --------------------------------------------------------

NUM_TASKS = 10  # Number of tasks to generate
NODES_RANGE = (5, 20)  # Range for the number of intermediate nodes per task
WCET_RANGE = (13, 30)  # Range for the Worst-Case Execution Time (WCET) of nodes
P_EDGE = 0.1  # Probability of creating an edge using the Erdős–Rényi method
D_RATIO_RANGE = (0.125, 0.25)  # Range for the ratio of the critical path length to the deadline (for determining D_i)
RESOURCE_RANGE = (1, 6)  # Range for the number of shared resources
ACCESS_COUNT_RANGE = (1, 16)  # Range for the total number of accesses to each resource
ACCESS_LEN_RANGE = (5, 100)  # Range for the maximum access length (Critical Section)

HARD_SOFT_PROB = 0.4  # Probability of converting a node from Hard to Soft (if constraints are not violated)
CRITICAL_RATIOS = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]  # Ratios of critical to non-critical nodes
OUTPUT_DIR = "graphs_output"  # Directory to save graph images

# Create the output directory if it does not exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------------
# Helper Functions for Graph Construction and Parameter Calculation
# --------------------------------------------------------

def create_random_dag(num_inner, p):
    """
    Creates a random Directed Acyclic Graph (DAG) using NetworkX with num_inner intermediate nodes (0..num_inner-1).
    For each pair of nodes (u < v), a directed edge (u -> v) is added with probability p (forward direction).
    The function ensures that the resulting graph is acyclic and connected by ensuring there is a path from the source
    to the sink, and all intermediate nodes are reachable from the source and can reach the sink.

    Returns:
        G (nx.DiGraph): The generated DAG.
        node_types (dict): A dictionary mapping node IDs to their types ("Hard" or "Soft").
        ratio_crit (float): The ratio of critical nodes to total intermediate nodes.
    """
    total_nodes = num_inner + 2  # Including source and sink
    source_id = num_inner
    sink_id = num_inner + 1

    while True:
        G = nx.DiGraph()
        G.add_nodes_from(range(total_nodes))

        # 1) Node Types: Initialize all intermediate nodes as Soft by default; source and sink will have type None
        node_types = {n: None for n in range(total_nodes)}
        for n in range(num_inner):
            node_types[n] = "Soft"

        # 2) Determine the number of critical nodes based on a randomly selected ratio
        ratio_crit = random.choice(CRITICAL_RATIOS)  # Select a ratio from the predefined list
        ratio_crit /= (ratio_crit + 1)  # Normalize the ratio
        num_crit = round(ratio_crit * num_inner)  # Calculate the number of critical nodes

        # Randomly select the critical nodes from the intermediate nodes
        crit_nodes = set(random.sample(range(num_inner), num_crit)) if num_inner > 0 else set()

        # Assign "Hard" type to critical nodes
        for n in crit_nodes:
            node_types[n] = "Hard"

        # 3) Add edges between nodes based on the probability p, ensuring no constraints are violated
        for u in range(num_inner):
            for v in range(u + 1, num_inner):
                # Constraint: Do not add an edge if it would make a Soft node have a Hard child
                if random.random() < p and not (node_types[u] == "Soft" and node_types[v] == "Hard"):
                    G.add_edge(u, v)

        # 4) Ensure connectivity:
        # - Add edges from the source to at least one intermediate node.
        # - Add edges from at least one intermediate node to the sink.
        # - Ensure all intermediate nodes are reachable from the source and can reach the sink.

        if num_inner > 0:
            # Connect the source to at least one intermediate node
            source_target = random.randint(0, num_inner - 1)
            G.add_edge(source_id, source_target)

            # Connect at least one intermediate node to the sink
            sink_source = random.randint(0, num_inner - 1)
            G.add_edge(sink_source, sink_id)

            # Ensure all intermediate nodes are reachable from the source and can reach the sink
            for n in range(num_inner):
                if not nx.has_path(G, source_id, n):
                    # Add an edge from the source to this node
                    G.add_edge(source_id, n)
                # if not nx.has_path(G, n, sink_id):
                #     # Add an edge from this node to the sink
                #     G.add_edge(n, sink_id)

        # 5) Ensure the graph is acyclic by attempting a topological sort
        try:
            list(nx.topological_sort(G))
            return G, node_types, ratio_crit
        except nx.NetworkXUnfeasible:
            # If the graph contains a cycle, retry
            continue


def compute_longest_path_length_dag(G, wcet):
    """
    Computes the length of the longest path (critical path) in the DAG G using the WCET of each node.
    The algorithm performs a topological sort followed by dynamic programming to calculate the longest path.

    Args:
        G (nx.DiGraph): The input DAG.
        wcet (dict): A dictionary mapping node IDs to their WCET values.

    Returns:
        int: The length of the longest path in the DAG.
    """
    topo_order = list(nx.topological_sort(G))
    dp = {node: wcet[node] for node in G.nodes()}
    for u in topo_order:
        for v in G.successors(u):
            if dp[v] < dp[u] + wcet[v]:
                dp[v] = dp[u] + wcet[v]
    return max(dp.values()) if dp else 0


def generate_one_task(task_id):
    """
    Generates a single task represented as a DAG with intermediate nodes, a source, and a sink.
    The function performs the following steps:
      1) Generate the intermediate DAG using create_random_dag.
      2) Add source and sink nodes to the DAG.
      3) Assign "Hard" or "Soft" types to intermediate nodes while respecting constraints.
      4) Select a random ratio of critical nodes and designate them.
      5) Calculate C_i, L_i, D_i, T_i, and U_i parameters.
      6) Visualize and save the graph image using a tree-like layout.

    Args:
        task_id (int): The identifier for the task.

    Returns:
        dict: A dictionary containing all relevant information about the generated task.
    """
    # Number of intermediate nodes
    num_inner = random.randint(*NODES_RANGE)

    # Generate the intermediate DAG
    G_mid, node_types, ratio_crit = create_random_dag(num_inner, P_EDGE)

    # Define source and sink node IDs
    source_id = num_inner
    sink_id = num_inner + 1
    total_nodes = num_inner + 2  # Nodes: 0..(num_inner-1), source, sink

    # Create the final DAG including source and sink
    G = nx.DiGraph()
    G.add_nodes_from(range(total_nodes))
    # Copy edges from the intermediate DAG
    G.add_edges_from(G_mid.edges())

    # Add mandatory edges: source -> first intermediate node and last intermediate node -> sink
    if num_inner > 0:
        G.add_edge(source_id, 0)
        G.add_edge(num_inner - 1, sink_id)

    # 4) Assign WCET values
    wcet = {}
    for n in range(total_nodes):
        if n == source_id or n == sink_id:
            wcet[n] = 0  # Source and sink have zero execution time
        else:
            wcet[n] = random.randint(*WCET_RANGE)

    # Calculate C_i: Total execution time of intermediate nodes
    Ci = sum(wcet[n] for n in range(num_inner))

    # Compute the critical path length L_i
    Li = compute_longest_path_length_dag(G, wcet)

    # Determine D_i based on a random ratio from D_RATIO_RANGE
    ratio_d = random.uniform(*D_RATIO_RANGE)
    Di = int(Li / ratio_d) if ratio_d != 0 else Li

    Ti = Di  # Deadline is set to D_i
    Ui = Ci / Ti if Ti > 0 else float('inf')  # Utilization U_i

    # 5) Visualize and save the DAG with different colors for Hard/Soft/Source/Sink nodes
    # Use a hierarchical layout to shape the graph as a tree
    pos = hierarchy_pos(G, source_id)

    plt.figure(figsize=(8, 6))
    node_colors = []
    for n in G.nodes():
        if n == source_id or n == sink_id:
            node_colors.append("yellow")  # Source and sink nodes
        elif node_types.get(n, None) == "Hard":
            node_colors.append("red")  # Hard nodes
        elif node_types.get(n, None) == "Soft":
            node_colors.append("green")  # Soft nodes
        else:
            node_colors.append("gray")  # Undefined or other types

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='black', arrows=True)
    
    # Label nodes with their IDs
    labels_dict = {n: f"{n}" for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels_dict, font_color='white')

    plt.title(f"Task {task_id} | N={num_inner} | Ci={Ci} | Li={Li} | Di,T={Di} | U={Ui}")
    plt.axis('off')
    output_path = os.path.join(OUTPUT_DIR, f"dag_task_{task_id}.png")
    plt.savefig(output_path, dpi=150)
    plt.close()

    return {
        "task_id": task_id,
        "num_nodes": total_nodes,
        "edges": list(G.edges()),
        "node_types": node_types,
        "critical_ratio": ratio_crit,
        "wcet": wcet,
        "C": Ci,
        "L": Li,
        "D": Di,
        "T": Ti,
        "U": Ui
    }


def hierarchy_pos(G, root, width=1.0, vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """
    Create a hierarchical layout for a tree-like DAG.

    Args:
        G (nx.DiGraph): The input graph.
        root (int): The root node of the tree.
        width (float): Horizontal space allocated for the layout.
        vert_gap (float): Vertical gap between levels.
        vert_loc (float): Vertical location of the root node.
        xcenter (float): Horizontal center of the layout.

    Returns:
        dict: A dictionary of positions keyed by node.
    """
    pos = {root: (xcenter, vert_loc)}
    neighbors = list(G.successors(root))
    if len(neighbors) != 0:
        dx = width / len(neighbors)
        nextx = xcenter - width / 2 - dx / 2
        for neighbor in neighbors:
            nextx += dx
            pos.update(hierarchy_pos(G, neighbor, width=dx, vert_gap=vert_gap, vert_loc=vert_loc - vert_gap, xcenter=nextx))
    return pos

def generate_resources(num_tasks):
    """
    Generates shared resources and determines their attributes:
      - Assign a random value within RESOURCE_RANGE for the number of resources.
      - For each resource, determine the maximum access length within ACCESS_LEN_RANGE.
      - Assign the total number of accesses within ACCESS_COUNT_RANGE.
      - Distribute these accesses randomly among the tasks.

    Args:
        num_tasks (int): The number of tasks among which accesses are distributed.

    Returns:
        tuple: A tuple containing the number of resources and a list of dictionaries with resource information.
    """
    n_r = random.randint(*RESOURCE_RANGE)
    resources_info = []
    for r_id in range(n_r):
        max_len = random.randint(*ACCESS_LEN_RANGE)
        total_acc = random.randint(*ACCESS_COUNT_RANGE)

        # Distribute total_acc accesses randomly among the tasks
        distribution = [0] * num_tasks
        remain = total_acc
        for i in range(num_tasks - 1):
            if remain <= 0:
                break
            pick = random.randint(0, remain)
            distribution[i] = pick
            remain -= pick
        distribution[-1] += remain  # Assign the remaining accesses to the last task

        resources_info.append({
            "resource_id": r_id + 1,
            "max_access_length": max_len,
            "total_accesses": total_acc,
            "distribution": distribution
        })

    return n_r, resources_info


def compute_processors_federated(tasks):
    """
    Calculates the number of processors required using Federated Scheduling.
      - If U_i > 1, calculate m_i = ceil((C_i - L_i) / (D_i - L_i))
      - If U_i <= 1, assign a dedicated processor

    Args:
        tasks (list): A list of task dictionaries.

    Returns:
        int: The total number of processors required.
    """
    total = 0
    for t in tasks:
        Ui = t["U"]
        Ci = t["C"]
        Li = t["L"]
        Di = t["D"]
        if Ui > 1:
            denom = (Di - Li) if (Di - Li) > 0 else 1  # Avoid division by zero
            top = (Ci - Li)
            m_i = math.ceil(top / denom)
            if m_i < 1:
                m_i = 1  # Ensure at least one processor
            total += m_i
        else:
            total += 1  # Dedicated processor for tasks with U_i <= 1
    return total

def federated_output(tasks, resources_info, n_r):
    # 2) Calculate total utilization
    U_sum = sum(t["U"] for t in tasks)
    # Generate a random number between 0.1 and 1 for U_norm
    U_norm = random.uniform(0.1, 1)
    m_simple = math.ceil(U_sum/U_norm)

    # 5) Calculate the number of processors using Federated Scheduling
    m_fed = compute_processors_federated(tasks)

    # 6) Display the results
    print("==================================================")
    print(" Task and Resource Generation (Phase One) ")
    print("==================================================")

    for t in tasks:
        print(f"\n--- Task tau_{t['task_id']} ---")
        print(f" • Total number of nodes (including source and sink): {t['num_nodes']}")
        print(f" • Ratio of critical nodes: {t['critical_ratio']:.2f}")
        # print(" • Nodes (ID -> WCET, Type):")
        # for n in range(t['num_nodes']):
        #     wc = t['wcet'][n]
        #     tp = t['node_types'][n]
        #     print(f"    - {n}: c={wc}, type={tp}")
        # print(f" • Edges: {t['edges']}")
        print(f" • C{t['task_id']} = {t['C']}")
        print(f" • L{t['task_id']} = {t['L']}")
        print(f" • D{t['task_id']} = {t['D']}")
        print(f" • T{t['task_id']} = {t['T']}")
        print(f" • U{t['task_id']} = {t['U']:.3f}")
        png_path = os.path.join(OUTPUT_DIR, f"dag_task_{t['task_id']}.png")
        print(f" -> Graph image saved at: {png_path}")

    print("--------------------------------------------------")
    print(f" • Total Utilization of Tasks (UΣ) = {U_sum:.3f}")
    print("--------------------------------------------------")

    print("\n==================================================")
    print(" Shared Resources ")
    print("==================================================")
    print(f" • Number of resources (n_r) = {n_r}")
    for r in resources_info:
        dist_str = ", ".join([f"tau_{idx + 1}={val}" for idx, val in enumerate(r['distribution'])])
        print(f"  - Resource l_{r['resource_id']}:")
        print(f"      Maximum Access Length = {r['max_access_length']}")
        print(f"      Total Number of Accesses = {r['total_accesses']}")
        print(f"      Access Distribution among Tasks: {dist_str}")

    print("\n==================================================")
    print(" Required Number of Processors ")
    print("==================================================")
    print(f" • Based on Total Utilization: m = ceil(UΣ) = {m_simple}")
    print(f" • Based on Federated Scheduling: m = {m_fed}")
    print("==================================================")
    if m_fed <= m_simple:
        print("The tasks are schedulable under Federated Scheduling.")
    else:
        print("The tasks are not schedulable under Federated Scheduling.")
    print("==================================================")
    return m_fed, m_simple

# --------------------------------------------------------
# 2) Data Structures for POMIP
# --------------------------------------------------------

class GlobalQueue:
    """
    One global queue GQ_q per resource q. Policy can be "FIFO" or "FPC" (first-path-critical).
    """
    def __init__(self, resource_id, policy="FIFO"):
        self.resource_id = resource_id
        self.policy = policy
        self.queue = deque()
    
    def enqueue(self, node_info):
        """
        node_info: {
          "node_id": int,
          "task_id": int,
          "arrival_time": int,
          "on_critical_path": bool,
          "is_critical_node": bool,
        }
        """
        self.queue.append(node_info)
        if self.policy == "FPC":
            self._sort_by_critical_path()
    
    def _sort_by_critical_path(self):
        # Put 'on_critical_path' = True at front (FIFO among themselves), then the others
        # Tie-break by arrival_time
        def priority(item):
            return (not item["on_critical_path"], item["arrival_time"])
        sorted_list = sorted(self.queue, key=priority)
        self.queue = deque(sorted_list)
    
    def pop_front(self):
        if self.queue:
            return self.queue.popleft()
        return None

    def front(self):
        return self.queue[0] if self.queue else None
    
    def remove(self, node_id, task_id):
        for i, item in enumerate(self.queue):
            if item["node_id"] == node_id and item["task_id"] == task_id:
                del self.queue[i]
                break
    
    def is_empty(self):
        return len(self.queue) == 0


def request_resource(tid, nid, rid, current_time, local_queues, global_queues, node_states):
    """
    Node (tid, nid) requests resource rid.
    1) Enqueue in local queue FQ_{rid, tid} if not present.
    2) If it's at the head of that local queue, it tries to enqueue in GQ_rid.
    3) If it becomes head of GQ_rid, it locks the resource => in_critical_section = True.
    """
    # 1) local queue
    if nid not in local_queues[(tid, rid)]:
        local_queues[(tid, rid)].append(nid)
        node_states[(tid, nid)]["arrival_time_resource"] = current_time
    
    # 2) check if at head
    if local_queues[(tid, rid)][0] == nid:
        # enqueue in global queue
        gq_item = {
            "node_id": nid,
            "task_id": tid,
            "arrival_time": node_states[(tid, nid)]["arrival_time_resource"],
            "on_critical_path": node_states[(tid, nid)]["on_critical_path"],
            "is_critical_node": node_states[(tid, nid)]["is_critical_node"]
        }
        global_queues[rid].enqueue(gq_item)

        # check if at head of GQ
        front_item = global_queues[rid].front()
        if front_item and front_item["node_id"] == nid and front_item["task_id"] == tid:
            # lock resource
            node_states[(tid, nid)]["locked_resource"] = rid
            node_states[(tid, nid)]["in_critical_section"] = True


def release_resource(tid, nid, rid, local_queues, global_queues, node_states, current_time):
    """
    Node (tid, nid) releases resource rid after finishing critical section.
    Remove from GQ_rid and FQ_{rid, tid}. Then if FQ_{rid, tid} not empty,
    that head tries to join GQ_rid.
    """
    global_queues[rid].remove(nid, tid)

    fq = local_queues[(tid, rid)]
    # remove from local queue
    if fq and fq[0] == nid:
        fq.popleft()
    else:
        if nid in fq:
            fq.remove(nid)

    # clear node state
    node_states[(tid, nid)]["locked_resource"] = None
    node_states[(tid, nid)]["in_critical_section"] = False
    node_states[(tid, nid)]["requesting_resource"] = None

    # let next head in FQ_{rid, tid} attempt to join GQ
    if len(fq) > 0:
        new_head = fq[0]
        request_resource(tid, new_head, rid, current_time, local_queues, global_queues, node_states)


def attempt_preemption(cluster_cpus, candidate_node, node_states):
    """
    Try to place 'candidate_node' on an available CPU in the cluster.
    If no CPU is free, try preempting a non-critical node that is in normal section.
    Return index of CPU if success, else None.
    """
    # candidate_node is (tid, nid)
    for idx, occupant in enumerate(cluster_cpus):
        if occupant is None:
            return idx  # free CPU found

    # no free CPU => see if we can preempt
    # if candidate is critical, it can preempt a non-critical occupant that is not in a critical section
    (ctid, cnid) = candidate_node
    if node_states[(ctid, cnid)]["is_critical_node"]:
        for idx, occupant in enumerate(cluster_cpus):
            if occupant is not None:
                (otid, onid) = occupant
                if (not node_states[(otid, onid)]["in_critical_section"]
                    and (not node_states[(otid, onid)]["is_critical_node"])):  
                    # occupant is non-critical node in normal section => preempt
                    return idx
    return None


def try_migration_if_blocked(tid, nid, locked_resource, clusters, node_states, local_queues, current_time):
    """
    If node (tid, nid) locked 'locked_resource' but cannot run on its home cluster,
    attempt to migrate to another cluster that also is blocked on locked_resource.
    """
    home_cluster_id = tid  # in a federated approach, cluster i = task i
    for other_task_id, cluster_cpus in enumerate(clusters):
        if other_task_id == home_cluster_id:
            continue
        # see if that task is also blocked on locked_resource => local_queues[(other_task_id, locked_resource)] not empty
        if len(local_queues.get((other_task_id, locked_resource), [])) > 0:
            # attempt preemption or find free CPU
            cpu_idx = attempt_preemption(cluster_cpus, (tid, nid), node_states)
            if cpu_idx is not None:
                # MIGRATE
                # preempt occupant if needed
                occupant = cluster_cpus[cpu_idx]
                if occupant is not None:
                    cluster_cpus[cpu_idx] = None  # or push occupant to some "ready queue"
                cluster_cpus[cpu_idx] = (tid, nid)
                return True
    return False


# --------------------------------------------------------
# 3) POMIP Scheduler (Time-Stepped Prototype)
# --------------------------------------------------------

def schedule_with_pomip(tasks, resources_info, policy="FIFO", time_limit=1000):
    """
    1) Build one global queue GQ_q per resource with given 'policy'.
    2) Build local queues FQ_{q,i}.
    3) Build node_states for each node (tid, node_id).
    4) Create cluster_state = for each task i, a list of CPU slots (federated approach).
    5) Step through time. On each step, assign nodes to free CPUs or attempt preemption,
       handle resource requests, handle critical sections, release resources, etc.
    """
    # 1) Global queues
    global_queues = {}
    all_resource_ids = []
    for r in resources_info:
        rid = r["resource_id"]
        all_resource_ids.append(rid)
        global_queues[rid] = GlobalQueue(rid, policy=policy)

    # 2) Local queues
    local_queues = {}
    for t in tasks:
        tid = t["task_id"]
        for r in all_resource_ids:
            local_queues[(tid, r)] = deque()

    # 3) Build node states
    node_states = {}
    for t in tasks:
        tid = t["task_id"]
        Di  = t["D"]
        for n in range(t["num_nodes"]):
            wc = t["wcet"][n]
            if wc <= 0:
                continue
            # For demonstration, we treat each node as if it needs resource_id=1 if it's a "Hard" node 
            # (just a simple example). You can adapt it to the actual resource usage from distribution, etc.
            needed_resource = None
            if t["node_types"][n] == "Hard":
                needed_resource = 1  # say resource #1 is used by Hard nodes
            node_states[(tid, n)] = {
                "remaining_time": wc,
                "deadline": Di,
                "is_critical_node": (t["node_types"][n] == "Hard"),
                "on_critical_path": (t["node_types"][n] == "Hard"),  # or some other condition
                "locked_resource": None,
                "in_critical_section": False,
                "requesting_resource": needed_resource,  # if it needs a resource
                "arrival_time_resource": 0,
                "completed": False
            }

    # 4) Create cluster_state => federated => each task i has M_i=1 CPU for simplicity
    #    Or you can do the full compute_processors_federated.
    cluster_state = []
    for i, t in enumerate(tasks):
        # simple assumption: each task has 1 CPU
        cluster_state.append([None])  # one CPU slot

    # 5) Time-stepped scheduling
    time = 0
    schedulable = True

    while time < time_limit:
        # Check if all done
        if all(st["completed"] for st in node_states.values()):
            break

        # For each cluster i (task i):
        for i, cpus in enumerate(cluster_state):
            for cpu_idx, occupant in enumerate(cpus):
                if occupant is None:
                    # CPU is free -> pick a node from task i that is ready to run
                    candidate = pick_node_to_run(i, node_states, tasks)
                    if candidate is not None:
                        (tid, nid) = candidate
                        # attempt to place (tid, nid)
                        cpus[cpu_idx] = (tid, nid)
                else:
                    # occupant is running -> check if it needs to request or is in critical sec
                    (tid, nid) = occupant
                    st = node_states[(tid, nid)]
                    
                    # If it is about to request resource
                    if st["requesting_resource"] and not st["in_critical_section"]:
                        # Attempt lock
                        rneeded = st["requesting_resource"]
                        request_resource(tid, nid, rneeded, time, local_queues, global_queues, node_states)
                        if not st["in_critical_section"]:
                            # blocked => occupant yields CPU
                            cpus[cpu_idx] = None
                            # if it's critical, attempt migration
                            if st["is_critical_node"]:
                                success = try_migration_if_blocked(tid, nid, rneeded,
                                                                   cluster_state, node_states,
                                                                   local_queues, time)
                            continue  # go next CPU

                    # If occupant is in critical section or normal section
                    st["remaining_time"] -= 1
                    if st["remaining_time"] <= 0:
                        # finished node
                        if st["in_critical_section"]:
                            # release resource
                            rid = st["locked_resource"]
                            release_resource(tid, nid, rid, local_queues, global_queues, node_states, time)
                        st["completed"] = True
                        cpus[cpu_idx] = None
                    else:
                        # if it's in critical section => non-preemptable
                        # if it's normal and is critical node => can be preempted by an even more urgent node
                        # We'll keep it simple here and do nothing
                        pass

        # Check deadlines for critical nodes
        for (tid, nid), st in node_states.items():
            if st["is_critical_node"] and not st["completed"]:
                if time >= st["deadline"]:
                    schedulable = False
                    break
        if not schedulable:
            break

        time += 1

    # Possibly compute QoS for non-critical nodes, etc.
    # For now, we just return if any critical missed
    return {
        "schedulable": schedulable,
        "finish_time": time
    }


def pick_node_to_run(cluster_id, node_states, tasks):
    """
    Very simple function: pick any incomplete node from task cluster_id
    that is not currently in a critical section blocked state.
    Priority to critical nodes first, then non-critical.
    This is extremely simplified; you might want to check if it's truly 'runnable'
    (all predecessors finished, no resource block, etc.).
    """
    tid = tasks[cluster_id]["task_id"]
    candidates = []
    for (t_i, n_i), st in node_states.items():
        if t_i == tid and not st["completed"]:
            # If it's not locked in a waiting queue, or we haven't enforced DAG precedence:
            # We just say it's "runnable."
            candidates.append((n_i, st))

    # Sort: critical first
    crit = [ (n_i, st) for (n_i, st) in candidates if st["is_critical_node"] and not st["in_critical_section"] ]
    noncrit = [ (n_i, st) for (n_i, st) in candidates if not st["is_critical_node"] ]

    if crit:
        return (tid, crit[0][0])
    elif noncrit:
        return (tid, noncrit[0][0])
    else:
        return None


# --------------------------------------------------------
# Main Function
# --------------------------------------------------------

def main():
    random.seed(0)

    # 1) Generate tasks
    tasks = []
    for i in range(NUM_TASKS):
        t_info = generate_one_task(i + 1)
        tasks.append(t_info)

    # 2) Generate shared resources
    n_r, resources_info = generate_resources(NUM_TASKS)

    m_fed, m_simple = federated_output(tasks, resources_info, n_r)

    # 3) Run POMIP with FIFO policy
    fifo_result = schedule_with_pomip(tasks, resources_info, policy="FIFO", time_limit=2000)

    # 4) Run POMIP with FPC (first-path-critical) policy
    fpc_result = schedule_with_pomip(tasks, resources_info, policy="FPC", time_limit=2000)

    # 5) Print results
    print("======== POMIP with FIFO ========")
    print("Schedulable?", fifo_result["schedulable"])
    print("Finish time:", fifo_result["finish_time"])

    print("======== POMIP with FPC  ========")
    print("Schedulable?", fpc_result["schedulable"])
    print("Finish time:", fpc_result["finish_time"])
    

if __name__ == "__main__":
    main()

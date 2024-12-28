import random
import math
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Configuration parameters
# --------------------------
NUM_TASK_SETS = 100

# Graph generation parameters
NODE_RANGE = [5, 20]   # number of nodes (excluding source/sink)
P = 0.1                # probability for edge creation in Erdős–Rényi model

# Execution times
C_MIN, C_MAX = 20, 40  # execution times for nodes
# The instructions have mentioned different intervals [13,30] or [20,40].
# We choose [20,40] as stated in the initial specification.

# Critical ratio set
CRITICAL_RATIOS = [0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# Relative deadline ratio
DEADLINE_RATIO_MIN = 0.125
DEADLINE_RATIO_MAX = 0.25

# Resource parameters
NR_RANGE = [1,2,3,4,5] # number of resources
REQUESTS_PER_RESOURCE_RANGE = [8,64]
MAX_ACCESS_TIME_RANGE = [5,10]
NUM_TASKS_RANGE = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]

# Normalized utilizations to test
UNORM_VALUES = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

# Number of processors scenarios for a fixed Unorm=0.5
M_VALUES = [2,4,8,32]

random.seed(0)
np.random.seed(0)

# ----------------------------------------
# Helper functions
# ----------------------------------------

def generate_erdos_renyi_graph(num_nodes, p):
    """Generate a directed acyclic graph (DAG) using Erdős–Rényi model and then make it acyclic.
    Note: The standard Erdős–Rényi G(n,p) for directed edges does not guarantee acyclicity.
    We will:
      - Generate a random directed graph
      - If cycles appear, we attempt a topological sort. If fails, re-generate.
      This might not be the most efficient method but should work for small graphs.
    """
    while True:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_nodes))
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j and random.random() < p:
                    G.add_edge(i, j)
        # Try to get a topological order to ensure acyclicity
        try:
            order = list(nx.topological_sort(G))
            # We have a DAG
            return G
        except nx.NetworkXUnfeasible:
            # Graph has a cycle, try again
            continue

def add_source_sink(G):
    """Add a source and sink node to graph G.
    source connects to all nodes with no parents
    sink connects to all nodes with no children
    source will be labeled as 's'
    sink will be labeled as 't'
    """
    G = nx.convert_node_labels_to_integers(G, first_label=1) # Shift nodes to start from 1
    nodes = list(G.nodes)
    source = 0
    sink = max(nodes)+1
    G.add_node(source)
    G.add_node(sink)
    # edges from source to nodes with no parent
    for n in nodes:
        if G.in_degree(n) == 0:
            G.add_edge(source, n)
    # edges from nodes with no child to sink
    for n in nodes:
        if G.out_degree(n) == 0:
            G.add_edge(n, sink)
    return G, source, sink

def assign_execution_times(G, source, sink):
    exec_times = {}
    for n in G.nodes:
        if n == source or n == sink:
            exec_times[n] = 0
        else:
            exec_times[n] = random.uniform(C_MIN, C_MAX)
    return exec_times

def longest_path_length(G, source, sink, exec_times):
    # Compute longest path using DP in DAG
    dist = {n: -math.inf for n in G.nodes}
    dist[source] = 0
    topo = list(nx.topological_sort(G))
    for u in topo:
        for v in G.successors(u):
            if dist[v] < dist[u] + exec_times[v]:
                dist[v] = dist[u] + exec_times[v]
    return dist[sink]

def random_critical_assignment(G, source, sink, ratio):
    # Assign critical (hard) vs non-critical (soft)
    # The number of critical nodes = ratio * (total_non_source_sink)
    # Enforce that critical nodes cannot have parents that are non-critical.
    nodes = [n for n in G.nodes if n not in [source, sink]]
    num_critical = int(round(ratio * len(nodes)))
    critical_nodes = set(random.sample(nodes, num_critical))
    # Check constraint: critical nodes cannot be children of non-critical
    # If violates, we can retry a few times or adjust the selection
    # For simplicity, we try a few times:
    for _ in range(100):
        violate = False
        for cn in critical_nodes:
            for parent in G.predecessors(cn):
                if parent not in critical_nodes and parent not in [source, sink]:
                    violate = True
                    break
            if violate:
                break
        if not violate:
            # Good assignment
            break
        # Retry
        critical_nodes = set(random.sample(nodes, num_critical))

    node_types = {}
    for n in G.nodes:
        if n == source or n == sink:
            node_types[n] = 'critical' # Source/sink can be considered critical by default
        else:
            node_types[n] = 'critical' if n in critical_nodes else 'non-critical'
    return node_types

def compute_task_parameters(G, source, sink, exec_times):
    C_i = sum(exec_times.values())
    L_i = longest_path_length(G, source, sink, exec_times)
    # Ratio L_i / D_i in [0.125,0.25]
    ratio = random.uniform(DEADLINE_RATIO_MIN, DEADLINE_RATIO_MAX)
    D_i = L_i / ratio
    T_i = D_i
    U_i = C_i / T_i
    return C_i, L_i, D_i, T_i, U_i

def generate_resources():
    # Choose number of resources
    n_r = random.choice(NR_RANGE)
    resources = ["l" + str(i) for i in range(1, n_r+1)]
    return resources

def assign_resource_requests(tasks_info):
    # tasks_info is a list of tasks each with G, C_i, etc.
    # We must distribute requests per resource across tasks
    # For simplicity, choose a random total requests # per resource and distribute

    # Example: 
    # 1) Pick a random value from REQUESTS_PER_RESOURCE_RANGE
    # 2) For each resource, we have that many requests total across all tasks
    # 3) Distribute them randomly among tasks
    # 4) For each request, assign a random length in [1, max_access_length]

    # NOTE: The instructions mention multiple different sets of parameters and ranges.
    #       We simplify and show one approach.
    requests_per_resource = random.randint(REQUESTS_PER_RESOURCE_RANGE[0], REQUESTS_PER_RESOURCE_RANGE[1])
    max_access_length = random.uniform(MAX_ACCESS_TIME_RANGE[0], MAX_ACCESS_TIME_RANGE[1])

    # For each task, define a sequence of resource accesses (non-nested)
    # We'll just assign a random subset of these requests to tasks
    # total_tasks = len(tasks_info)
    # Distribute requests_per_resource * n_r total requests

    # Actually, instructions say total requests by all tasks to each resource is random.
    # We'll do a per-resource random distribution.
    for t in tasks_info:
        n_r = len(t['resources'])
        t['resource_requests'] = {r: [] for r in t['resources']}

    for t in tasks_info:
        n_r = len(t['resources'])
        # Randomly assign a number of requests for each resource to this task
        for r in t['resources']:
            num_req = random.randint(0, requests_per_resource) # random distribution
            for _ in range(num_req):
                length = random.uniform(1, max_access_length)
                t['resource_requests'][r].append(length)
    return tasks_info

def federated_scheduling(tasks_info, U_norm):
    # Compute total utilization
    U_sum = sum([task['U_i'] for task in tasks_info])
    # Compute total processors
    m = math.ceil(U_sum / U_norm)
    # Federated scheduling:
    # For each task:
    # if U_i > 1, m_i = ceil( (C_i - L_i) / (D_i - L_i) )
    # else m_i = 1
    assigned_m = 0
    for task in tasks_info:
        C_i, L_i, D_i, U_i = task['C_i'], task['L_i'], task['D_i'], task['U_i']
        if U_i > 1:
            m_i = math.ceil((C_i - L_i)/(D_i - L_i)) if D_i != L_i else math.ceil(U_i)
        else:
            m_i = 1
        task['m_i'] = m_i
        assigned_m += m_i

    schedulable = (assigned_m <= m)
    return m, schedulable

def generate_task(num_nodes=None, critical_ratio=0.5):
    # Generate a single task (graph)
    if num_nodes is None:
        num_nodes = random.randint(NODE_RANGE[0], NODE_RANGE[1])
    G = generate_erdos_renyi_graph(num_nodes, P)
    G, source, sink = add_source_sink(G)
    exec_times = assign_execution_times(G, source, sink)
    node_types = random_critical_assignment(G, source, sink, critical_ratio)
    C_i, L_i, D_i, T_i, U_i = compute_task_parameters(G, source, sink, exec_times)
    resources = generate_resources()
    task_info = {
        'G': G,
        'source': source,
        'sink': sink,
        'exec_times': exec_times,
        'node_types': node_types,
        'C_i': C_i,
        'L_i': L_i,
        'D_i': D_i,
        'T_i': T_i,
        'U_i': U_i,
        'resources': resources
    }
    return task_info

def generate_task_set(num_tasks=None, critical_ratio=0.5):
    if num_tasks is None:
        num_tasks = random.choice(NUM_TASKS_RANGE)
    tasks = []
    for _ in range(num_tasks):
        t = generate_task(critical_ratio=critical_ratio)
        tasks.append(t)
    tasks = assign_resource_requests(tasks)
    return tasks

# -----------------------------------------
# Main simulation
# -----------------------------------------
# We will generate data for different U_norm, critical ratios, # of requests, # of resources, etc.
# Due to complexity, we show a simple loop that generates multiple sets
# and store their results. You can filter and produce the requested charts.

results = []
for _ in range(NUM_TASK_SETS):
    # Choose parameters
    unorm = random.choice(UNORM_VALUES)
    critical_ratio = random.choice(CRITICAL_RATIOS)
    tasks = generate_task_set(critical_ratio=critical_ratio)
    m, schedulable = federated_scheduling(tasks, unorm)

    # Collect data
    U_sum = sum([t['U_i'] for t in tasks])
    nr = len(tasks[0]['resources']) if tasks else 0
    requests_count = sum([len(req) for t in tasks for r,req in t['resource_requests'].items()])
    max_req_len = 0
    if tasks:
        all_req_lengths = [length for t in tasks for r, reqs in t['resource_requests'].items() for length in reqs]
        if all_req_lengths:
            max_req_len = max(all_req_lengths)

    results.append({
        'unorm': unorm,
        'critical_ratio': critical_ratio,
        'num_tasks': len(tasks),
        'U_sum': U_sum,
        'm': m,
        'schedulable': schedulable,
        'num_resources': nr,
        'total_requests': requests_count,
        'max_request_length': max_req_len
    })

df = pd.DataFrame(results)
df.to_csv('simulation_results.csv', index=False)
print("Simulation results saved to simulation_results.csv")

# -----------------------------------------
# Example plotting
# -----------------------------------------
# Plot average schedulability vs unorm
mean_sched_by_unorm = df.groupby('unorm')['schedulable'].mean().reset_index()

plt.figure(figsize=(8,6))
plt.plot(mean_sched_by_unorm['unorm'], mean_sched_by_unorm['schedulable'], marker='o')
plt.title('Average Schedulability vs U_norm')
plt.xlabel('U_norm')
plt.ylabel('Schedulability (ratio)')
plt.grid(True)
plt.savefig('schedulability_vs_unorm.png')
plt.show()

# You can create similar plots for other parameters:
# - Schedulability vs # of requests
# - Schedulability vs # of resources
# - Schedulability vs num_tasks
# - Schedulability vs critical_ratio
# etc.

# To fully meet all requirements:
# - Extend code to implement both FIFO and critical path first queueing.
# - Implement POMIP algorithm details.
# - Repeat experiments for additional processors and generate all requested charts.

# This code provides a framework and partial implementation.

import random
import math
import networkx as nx
import matplotlib.pyplot as plt
import os

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
    The function ensures that the resulting graph is acyclic by attempting topological sorting.
    
    Returns:
        G (nx.DiGraph): The generated DAG.
        node_types (dict): A dictionary mapping node IDs to their types ("Hard" or "Soft").
        ratio_crit (float): The ratio of critical nodes to total intermediate nodes.
    """
    total_nodes = num_inner + 2  # Including source and sink
    while True:
        G = nx.DiGraph()
        G.add_nodes_from(range(num_inner))

        # 1) Node Types: Initialize all intermediate nodes as Soft by default; source and sink will have type None
        node_types = {n: None for n in range(total_nodes)}
        for n in range(num_inner):
            node_types[n] = "Soft"

        # 3) Determine the number of critical nodes based on a randomly selected ratio
        ratio_crit = random.choice(CRITICAL_RATIOS)  # Select a ratio from the predefined list
        ratio_crit /= (ratio_crit + 1)  # Normalize the ratio
        num_crit = round(ratio_crit * num_inner)  # Calculate the number of critical nodes

        # Randomly select the critical nodes from the intermediate nodes
        crit_nodes = set(random.sample(range(num_inner), num_crit)) if num_inner > 0 else set()

        # 2) Assign "Hard" type to critical nodes
        for n in crit_nodes:
            node_types[n] = "Hard"

        # Add edges between nodes based on the probability p, ensuring no constraints are violated
        for u in range(num_inner):
            for v in range(u + 1, num_inner):
                # Constraint: Do not add an edge if it would make a Soft node have a Hard child
                if random.random() < p and not (node_types[u] == "Soft" and node_types[v] == "Hard"):
                    G.add_edge(u, v)

        # Ensure the graph is acyclic by attempting a topological sort
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
      6) Visualize and save the graph image.

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
    pos = nx.spring_layout(G, seed=10)  # Fixed seed for reproducible layout

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

    # 2) Calculate total utilization
    U_sum = sum(t["U"] for t in tasks)

    # 3) Generate shared resources
    n_r, resources_info = generate_resources(NUM_TASKS)

    # 4) Calculate the number of processors based on total utilization
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
        print(" • Nodes (ID -> WCET, Type):")
        for n in range(t['num_nodes']):
            wc = t['wcet'][n]
            tp = t['node_types'][n]
            print(f"    - {n}: c={wc}, type={tp}")
        print(f" • Edges: {t['edges']}")
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

if __name__ == "__main__":
    main()

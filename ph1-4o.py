import networkx as nx
import random
import math

# Parameters
NUM_TASKS = 5  # Number of tasks
P = 0.1  # Probability for Erdős-Rényi graph
NODE_EXEC_TIME_RANGE = (5, 20)
RELATIVE_DEADLINE_RANGE = (0.125, 0.25)
RESOURCES_RANGE = (1, 6)
ACCESS_COUNT_RANGE = (1, 16)
ACCESS_LENGTH_RANGE = (1, 5)
RESOURCE_ACCESS_TIME = (5, 100)

# Step 1: Task Graph Generation
def generate_task_graph():
    G = nx.erdos_renyi_graph(random.randint(5, 20), P, directed=True)
    nx.set_edge_attributes(G, 0, "weight")
    nx.set_node_attributes(G, 0, "execution_time")

    # Add source and sink nodes
    source = "source"
    sink = "sink"
    G.add_node(source, execution_time=0)
    G.add_node(sink, execution_time=0)

    for node in G.nodes:
        if node != source and node != sink:
            G.nodes[node]["execution_time"] = random.randint(*NODE_EXEC_TIME_RANGE)
            G.nodes[node]["type"] = random.choice(["hard", "soft"])

    return G

# Step 2: Resource Generation
def generate_resources():
    num_resources = random.randint(*RESOURCES_RANGE)
    return {f"resource_{i}": {"max_access": random.randint(*ACCESS_COUNT_RANGE)} for i in range(num_resources)}

# Step 3: Resource Allocation
def allocate_resources(G, resources):
    for node in G.nodes:
        if node not in ["source", "sink"]:
            resource = random.choice(list(resources.keys()))
            G.nodes[node]["resource"] = resource
            G.nodes[node]["access_time"] = random.randint(*ACCESS_LENGTH_RANGE)

# Step 4: Processor Count Calculation
def calculate_processors(task_graphs):
    total_utilization = sum(sum(nx.get_node_attributes(G, "execution_time").values()) for G in task_graphs)
    normalized_utilization = random.uniform(0.1, 1)
    return math.ceil(total_utilization / normalized_utilization)

# Step 5: Federated Scheduling
def federated_scheduling(task_graphs, num_processors):
    scheduling = {}
    for i, G in enumerate(task_graphs):
        utilization = sum(nx.get_node_attributes(G, "execution_time").values())
        if utilization > 1:
            scheduling[f"task_{i}"] = "exclusive"
        else:
            scheduling[f"task_{i}"] = "shared"
    return scheduling

# Main Workflow
def main():
    task_graphs = [generate_task_graph() for _ in range(NUM_TASKS)]
    resources = generate_resources()
    for G in task_graphs:
        allocate_resources(G, resources)

    num_processors = calculate_processors(task_graphs)
    scheduling = federated_scheduling(task_graphs, num_processors)

    print(f"Number of Processors: {num_processors}")
    print(f"Scheduling: {scheduling}")

if __name__ == "__main__":
    main()

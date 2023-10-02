import argparse
from collections import defaultdict
import os
import pickle
import numpy as np
import pandas as pd
from igraph import Graph


def convert_dataset_to_dynamic_graph(path, dataset_name="infectious_ct1"):
    # Load the dataset files
    A = pd.read_csv(f"{path}/{dataset_name}_A.txt", header=None).values
    graph_indicator = pd.read_csv(f"{path}/{dataset_name}_graph_indicator.txt", header=None).values.squeeze()
    graph_labels = pd.read_csv(f"{path}/{dataset_name}_graph_labels.txt", header=None).values.squeeze()
    node_labels = pd.read_csv(f"{path}/{dataset_name}_node_labels.txt", header=None, names=range(4), dtype=str)
    node_labels = node_labels.applymap(lambda x: pd.to_numeric(x, errors='coerce'))
#     node_labels = node_labels[0].str.split(',', expand=False)  # split each line into a list of strings
#     node_labels = node_labels.apply(lambda x: [int(i) for i in x])  # convert strings to integers
    edge_attributes = pd.read_csv(f"{path}/{dataset_name}_edge_attributes.txt", header=None).values.squeeze()
    # Get unique graph ids
    graph_ids = np.unique(graph_indicator)

    dynamic_graphs = defaultdict(list)
    labels = defaultdict(list)

    for graph_id in graph_ids:
        graph_nodes = np.where(graph_indicator == graph_id)[0] + 1  # node ids are 1-indexed
        graph_nodes_set = set(graph_nodes)  # convert to set for faster lookups
        
        # construct graph timeline
        graph_timeline = defaultdict(lambda: {"nodes": set(), "edges": set()})  # change "edges" to a set
        for node in graph_nodes:
            node_label_data = node_labels.loc[node-1]  # get the list of timestamps and labels for this node
            node_label_data_values = node_label_data.values  # Convert pandas series to numpy array
            for i in range(0, len(node_label_data_values) - 1, 2):  # iterate over pairs, avoid index out of range
                timestamp, label = node_label_data_values[i], node_label_data_values[i+1]
                if not np.isnan(timestamp):  # if the timestamp is not NaN
                    graph_timeline[timestamp]["nodes"].add((node, label))
                
        for i in range(len(A)):  # iterate over all edges in A
            if A[i, 0] in graph_nodes_set and A[i, 1] in graph_nodes_set:  # if the edge belongs to the current graph
                timestamp = edge_attributes[i]  # get the timestamp from edge_attributes
                graph_timeline[timestamp]["edges"].add((A[i, 0], A[i, 1]))  # add edge to graph_timeline

        # construct igraph graphs for each timestamp, keep edges from previous timestamps
        current_edges = set()
        node_dict = {}  # dictionary to keep track of node indices and their labels

        g = Graph()
        for timestamp, graph_data in sorted(graph_timeline.items()):
            # Add or update nodes and labels
            for node, label in graph_data["nodes"]:
                try:
                    existing_node_names = g.vs["name"]
                except KeyError:
                    existing_node_names = []
                if str(node) not in existing_node_names:
                    g.add_vertex(name=str(node))
                node_dict[node] = label
            
            # Update the igraph object
            for v in g.vs:
                node_name = int(v["name"])
                try:
                    existing_label = v["label"]
                except KeyError:
                    existing_label = None
                v["label"] = node_dict.get(node_name, existing_label)
            # Add edges
            current_edges = graph_data["edges"]
            g.add_edges([(str(u), str(v)) for (u, v) in current_edges])
            
            # Add graph label
            g["label"] = graph_labels[graph_id - 1]
            
            # Append a copy of the graph to dynamic_graphs
            dynamic_graphs[graph_id].append(g.copy())
            
            labels[graph_id] = graph_labels[graph_id - 1]

    return dynamic_graphs, labels

# TODO: this is replicated here and in generate_dynamic_graphs.py
# Save the dynamic graphs and their labels like in save_dynamic_graphs()
def save_dynamic_graphs(dynamic_graphs, labels, path):
    """
    Save dynamic graphs and their labels to a single pickle file.

    Parameters:
    graphs (list): A list of networkx graphs representing the dynamic graphs.
    labels (list): A list of labels for the dynamic graphs.
    filename (str): The name of the pickle file to save the dynamic graphs and labels to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for graph_id in dynamic_graphs.keys():
        graph = dynamic_graphs[graph_id]
        label = labels[graph_id]
        filename = os.path.join(path, f"dynamic_graph_{graph_id}.pkl")
        with open(filename, "wb") as f:
            pickle.dump((graph, label), f)

def main():
    parser = argparse.ArgumentParser(description='Convert and save dynamic graphs.')
    parser.add_argument('--dataset', type=str, default="infectious_ct1", help='Name of the dataset to convert')
    parser.add_argument('--path', type=str, help='Path to the dataset')

    args = parser.parse_args()

    # Generate the dynamic graphs
    dynamic_graphs, labels = convert_dataset_to_dynamic_graph(
        args.path, args.dataset
    )

    # Save the dynamic graphs
    save_dynamic_graphs(dynamic_graphs, labels, f"./data/labeled_datasets/{args.dataset}/")

if __name__ == "__main__":
    main()
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import random
import pickle


def generate_and_save_dag(args):
    """
    Generate a random Directed Acyclic Graph (DAG), find paths, and save them.

    Args:
        args: Command-line arguments containing parameters for DAG generation.
    """
    # Set the random seed for reproducibility
    random.seed(args.seed)

    # Generate a random directed graph using the G(n, p) model
    G = nx.gnp_random_graph(args.nodes, args.probability, directed=True)

    # Convert the graph to a weighted DAG (retain only edges where u < v)
    DAG = nx.DiGraph(
        [(u, v, {"weight": random.uniform(0.1, 1.0)}) for (u, v) in G.edges() if u < v]
    )

    # Assign labels to nodes based on degree (higher-degree nodes get higher labels)
    node_degrees = dict(DAG.degree())
    sorted_nodes = sorted(node_degrees, key=node_degrees.get, reverse=True)
    node_labels = {node: label for label, node in enumerate(sorted_nodes, start=1)}

    # Shuffle the labels if the `shuffle_labels` argument is provided
    if args.shuffle_labels:
        labels = list(node_labels.values())
        random.shuffle(labels)
        shuffled_labels = {node: labels[i] for i, node in enumerate(sorted_nodes)}
        node_labels = shuffled_labels
        # shuffle also the node ids
        shuffled_nodes = sorted(node_labels, key=node_labels.get)
        DAG = nx.relabel_nodes(
            DAG, {node: shuffled_nodes[i] for i, node in enumerate(sorted_nodes)}
        )
        node_labels = {
            node: label for label, node in enumerate(shuffled_nodes, start=1)
        }

    # Set the node labels as attributes in the DAG
    nx.set_node_attributes(DAG, node_labels, "label")

    # Reweight edges to favor connections to high-label nodes
    for u, v, data in DAG.edges(data=True):
        data["weight"] *= node_labels[v] / (node_labels[u] + node_labels[v])

    # Visualize the DAG if verbose mode is enabled
    if args.verbose:
        pos = nx.spring_layout(DAG)
        node_colors = [node_labels[node] for node in DAG.nodes()]
        nx.draw(
            DAG,
            pos,
            with_labels=True,
            node_color=node_colors,
            cmap=plt.cm.Blues,
            edge_color="gray",
        )
        plt.title("Generated DAG with Biased Labels")
        plt.show()

    # Ensure the generated graph is a DAG
    if not nx.is_directed_acyclic_graph(DAG):
        raise ValueError("Generated graph is not a DAG.")

    # Prepare for path generation
    one_length_paths = []  # One-length paths (direct edges)
    paths = []  # All paths of length > 2
    max_length = -1  # Track the maximum path length

    # Extract one-length paths (direct edges)
    for edge in DAG.edges():
        path = [edge[0], edge[1], edge[0], edge[1]]
        one_length_paths.append(path)

    # Find all simple paths between all pairs of nodes
    for source_node in DAG.nodes():
        for target_node in DAG.nodes():
            if source_node != target_node:
                all_paths = list(
                    nx.all_simple_paths(DAG, source=source_node, target=target_node)
                )
                if len(all_paths) > 0:
                    # Ensure diverse path lengths
                    direct_paths = [p for p in all_paths if len(p) == 2]
                    longer_paths = [p for p in all_paths if len(p) > 2]
                    sampled_direct = random.sample(
                        direct_paths, min(10, len(direct_paths))
                    )
                    sampled_longer = random.sample(
                        longer_paths, min(10, len(longer_paths))
                    )

                    for path in sampled_direct + sampled_longer:
                        formatted_path = [path[0], path[-1]] + path
                        if len(formatted_path) > max_length:
                            max_length = len(formatted_path)
                        paths.append(formatted_path)
                        #print(formatted_path)

    # Shuffle the paths to randomize the order
    random.shuffle(paths)

    # Split the paths into training and validation sets
    split_idx = int(args.ratio * len(paths))
    train_paths = paths[:split_idx] + one_length_paths
    random.shuffle(train_paths)
    val_paths = paths[split_idx:]

    # Save training paths to the specified file
    train_paths_filled = ""
    for path in train_paths:
        path_filled = f"{path[0]} {path[1]} " + " ".join(map(str, path[2:])) + " \n"
        train_paths_filled += path_filled

    with open(args.train_file, "w", encoding="utf8") as file:
        file.write(train_paths_filled)

    # Save validation paths to the specified file
    val_paths_filled = ""
    for path in val_paths:
        path_filled = f"{path[0]} {path[1]} " + " ".join(map(str, path[2:])) + " \n"
        val_paths_filled += path_filled

    with open(args.val_file, "w", encoding="utf8") as file:
        file.write(val_paths_filled)

    # Save the DAG as a pickle file
    with open(args.dag_file, "wb") as f:
        pickle.dump(DAG, f, pickle.HIGHEST_PROTOCOL)

    if args.verbose:
        print("DAG and paths successfully saved.")


def main():
    """
    Parse command-line arguments and execute the DAG generation process.
    """
    parser = argparse.ArgumentParser(
        description="Generate a random DAG and save paths."
    )

    # Add command-line arguments with default values
    parser.add_argument(
        "--nodes",
        type=int,
        default=100,
        help="Number of nodes in the DAG (default: 100).",
    )
    parser.add_argument(
        "--probability",
        type=float,
        default=0.1,
        help="Probability for edge creation (default: 0.1).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.8,
        help="Train/validation split ratio (default: 0.8).",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="../data/train_reasoning.txt",
        help="Output file for training paths.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="../data/val_reasoning.txt",
        help="Output file for validation paths.",
    )
    parser.add_argument(
        "--dag_file",
        type=str,
        default="../data/DAG.gpickle",
        help="Output file for the DAG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for detailed output.",
    )
    parser.add_argument(
        "--shuffle_labels",
        action="store_true",
        help="Shuffle node labels after assignment.",
    )

    # Parse arguments and call the DAG generation function
    args = parser.parse_args()
    generate_and_save_dag(args)


if __name__ == "__main__":
    main()

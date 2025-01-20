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
    # Weights can be used in future for mny tests
    DAG = nx.DiGraph(
        [(u, v, {"weight": random.randint(-10, 10)}) for (u, v) in G.edges() if u < v]
    )

    # Visualize the DAG if verbose mode is enabled
    if args.verbose:
        nx.draw(DAG, with_labels=True)
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
                    k = min(
                        20, len(all_paths)
                    )  # Min limit to 20 paths per source-target pair, this pram cames from the reference paper,
                    # it can be changed to any number.
                    for path in random.sample(all_paths, k):
                        if len(path) > 2:
                            if args.verbose:
                                formatted_path = " ".join(map(str, path))
                                print(f"Path: {formatted_path}")
                            path = [path[0], path[-1]] + path
                            if len(path) > max_length:
                                max_length = len(path)
                            paths.append(path)

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
        path_filled = path + ["\n"]
        train_paths_filled += " ".join(map(str, path_filled))

    with open(args.train_file, "w", encoding="utf8") as file:
        file.write(train_paths_filled)

    # Save validation paths to the specified file
    val_paths_filled = ""
    for path in val_paths:
        path_filled = path + ["\n"]
        val_paths_filled += " ".join(map(str, path_filled))

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
        default="data/train_reasoning.txt",
        help="Output file for training paths.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/val_reasoning.txt",
        help="Output file for validation paths.",
    )
    parser.add_argument(
        "--dag_file",
        type=str,
        default="data/DAG.gpickle",
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

    # Parse arguments and call the DAG generation function
    args = parser.parse_args()
    generate_and_save_dag(args)


if __name__ == "__main__":
    main()

# Example run:
# python run_data_generation.py --nodes 100 --probability 0.1 --ratio 0.9 --train_file data/train.txt --val_file data/val.txt --dag_file data/dag.gpickle --seed 123 --verbose

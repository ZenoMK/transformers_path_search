import argparse
import matplotlib.pyplot as plt
import networkx as nx
import pickle
import tiktoken

import torch
from utils_final import GPTModel, generate_and_print_sample, create_tuples, load_config


def load_model_and_data(
    config_file, train_file, val_file, dag_file, model_file, device
):
    """
    Load the model, tokenizer, and data necessary for validation.

    Args:
        config_file (str): Path to the configuration file.
        train_file (str): Path to the training file.
        val_file (str): Path to the validation file.
        dag_file (str): Path to the DAG file.
        model_file (str): Path to the model file.
        device (str): Device to load the model on.

    Returns:
        tuple: Model, tokenizer, device, validation paths, and DAG.
    """
    # Read the model configuration
    GPT_CONFIG = load_config(config_file)

    # Initialize the model and load pre-trained weights
    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(torch.load(model_file, map_location=device))

    # Set random seed for reproducibility
    torch.manual_seed(123)

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Read training and validation paths
    with open(train_file, "r", encoding="utf8") as file:
        train_paths = file.read()

    with open(val_file, "r", encoding="utf8") as file:
        val_paths = file.read()

    # Load the DAG from file
    with open(dag_file, "rb") as f:
        DAG = pickle.load(f)

    return model, tokenizer, device, val_paths, DAG


def validate_paths(val_paths, model, tokenizer, device, DAG, verbose=False):
    """
    Validates paths using a model and checks if they align with the expected DAG structure.

    Args:
        val_paths (str): Paths to validate, separated by newlines.
        model: The model used for generating path samples.
        tokenizer: The tokenizer used with the model.
        device: The device where the model operates.
        DAG: Directed Acyclic Graph for path validation.
        verbose (bool): If True, prints detailed information about the validation process.

    Returns:
        dict: A dictionary with counts of validation results.
    """
    counts = {"ok": 0, "not_ok": 0, "hallucination": 0, "not_reachable": 0}
    line_count = 0
    for path_line in val_paths.strip().split("\n"):
        # show line count for verbose mode
        if verbose:
            print(f"\n🔢 Line {line_count + 1}")
        path_parts = path_line.strip().split(" ")
        if len(path_parts) < 3:
            if verbose:
                print("❌ Invalid path format, skipping.")
            continue

        source_node = path_parts[0]
        target_node = path_parts[1]
        intermediate_nodes = path_parts[2:]

        if verbose:
            print(
                f"\n  Validating Path:  {source_node} ➡️ {target_node} (Intermediate: {', '.join(intermediate_nodes)})"
            )

        path_head = " ".join(path_parts[:3])  # Extract head of the path
        node_target = path_parts[1]  # Extract target node

        # Generate path sample using the model
        path_hat = generate_and_print_sample(
            model,
            tokenizer,
            device,
            path_head,
            max_new_tokens=60,
            print_output=False,
            proba_threshold=0.001,
        )

        # Process the generated emission path
        emission_path = path_hat.split("\n")[0].strip()  # Get first emission
        emission_tokens = emission_path.split(" ")[2:]  # Remove start tokens
        decorated_emission = (
            f" {source_node} ➡️ {target_node} (Emitted: {' ➡️ '.join(emission_tokens)})"
        )

        if verbose:
            print(f"🔍 Generated Path: {decorated_emission}")

        edges_to_check = create_tuples(emission_tokens)  # Create edges

        # Pre-check all edges against the DAG before proceeding
        has_hallucination = False
        for edge in edges_to_check:
            if not DAG.has_edge(int(edge[0]), int(edge[1])):
                if verbose:
                    print(f"❌ Pre-check: Edge {edge} not found in DAG.")
                counts["hallucination"] += 1
                has_hallucination = True

        if has_hallucination:
            counts["not_ok"] += 1
            continue

        if not edges_to_check or node_target not in emission_tokens:
            if verbose:
                print(
                    f"❌ Path Validation Failed: Final node {node_target} not reachable in emitted path."
                )
            counts["not_ok"] += 1
            counts["not_reachable"] += 1
            continue

        # Validate edges against the DAG
        is_valid = True
        for edge in edges_to_check:
            if edge[0] == node_target:
                if verbose:
                    print("✅ Node target found in emission path. OK.")
                break

            if not DAG.has_edge(int(edge[0]), int(edge[1])):
                if verbose:
                    print(f"❌ Path Validation Failed: Edge {edge} not found in DAG.")
                counts["not_ok"] += 1
                counts["hallucination"] += 1
                is_valid = False
                break

        if is_valid:
            if verbose:
                print(f"✅ Emission Path Validated Successfully.")
            counts["ok"] += 1
        line_count += 1

    # Calculate error percentage
    error_percentage = 100 * counts["not_ok"] / max(counts["ok"], 1)

    # Beautify final counts display
    print("\n✨ Validation Summary ✨")
    print("----------------------")
    print(f"✅ Successful Validations: {counts['ok']}")
    print(f"❌ Failed Validations: {counts['not_ok']}")
    print(f"🚫 Hallucinations: {counts['hallucination']}")
    print(f"🔴 Not Reachable: {counts['not_reachable']}")
    print(f"⚠️ Error Percentage: {error_percentage:.2f}%")
    print("----------------------")

    return counts


def main():
    """
    Parse command-line arguments and validate paths using a pre-trained GPT model.
    """
    parser = argparse.ArgumentParser(
        description="Validate paths using a pre-trained GPT model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="data/GPT_CONFIG_124M.pkl",
        help="Path to the model configuration file. Default is 'data/GPT_CONFIG_124M.pkl'.",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="data/train_reasoning.txt",
        help="Path to the training data file. Default is 'data/train_reasoning.txt'.",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="data/val_reasoning.txt",
        help="Path to the validation data file. Default is 'data/val_reasoning.txt'.",
    )
    parser.add_argument(
        "--dag_file",
        type=str,
        default="data/DAG.gpickle",
        help="Path to the DAG file. Default is 'data/DAG.gpickle'.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/path_finder.model",
        help="Path to the model file. Default is 'models/path_finder.model'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on. Options are 'cpu' or 'cuda'. Default is 'cpu'.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for detailed validation prints.",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # Load model and data, then validate paths
    model, tokenizer, device, val_paths, DAG = load_model_and_data(
        args.config_file,
        args.train_file,
        args.val_file,
        args.dag_file,
        args.model_file,
        device,
    )
    validate_paths(val_paths, model, tokenizer, device, DAG, verbose=args.verbose)


if __name__ == "__main__":
    main()

# Example run:
# python run_validation.py --config config/gpt_config.json --model_file models/path_finder.model --train_file data/train.txt --val_file data/val.txt --dag_file data/dag.gpickle --verbose

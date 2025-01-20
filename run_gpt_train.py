import argparse
import json
import matplotlib.pyplot as plt
import networkx as nx
import tiktoken

import torch
from torch.utils.data import Dataset, DataLoader
from utils_final import (
    GPTModel,
    create_dataloader_v1,
    train_model_simple,
    plot_losses,
)


def load_config(config_file):
    """
    Load GPT configuration from a JSON file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: GPT configuration.
    """
    with open(config_file, "r") as f:
        config = json.load(f)
    return config


def setup_config_and_data(train_file, val_file, config_file):
    """
    Set up GPT configuration, tokenizer, and load data from disk.

    Args:
        train_file (str): Path to the training data file.
        val_file (str): Path to the validation data file.
        config_file (str): Path to the GPT configuration file.

    Returns:
        tuple: GPT configuration, tokenizer, training DataLoader, and validation DataLoader.
    """
    GPT_CONFIG = load_config(config_file)

    # Initialize tokenizer using GPT-2 encoding
    tokenizer = tiktoken.get_encoding("gpt2")

    # Read training and validation data from files
    with open(train_file, "r", encoding="utf8") as file:
        train_paths = file.read()
    with open(val_file, "r", encoding="utf8") as file:
        val_paths = file.read()

    # Create DataLoader instances for training and validation datasets
    train_loader = create_dataloader_v1(
        train_paths,
        batch_size=GPT_CONFIG["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        drop_last=False,
        shuffle=True,
    )
    val_loader = create_dataloader_v1(
        val_paths,
        batch_size=GPT_CONFIG["batch_size"],
        max_length=GPT_CONFIG["context_length"],
        stride=GPT_CONFIG["context_length"],
        drop_last=False,
        shuffle=True,
    )

    return GPT_CONFIG, tokenizer, train_loader, val_loader


def train_and_save_model(
    GPT_CONFIG,
    tokenizer,
    train_loader,
    val_loader,
    device,
    num_epochs,
    verbose=False,
):
    """
    Train the GPT model and save it to disk.

    Args:
        GPT_CONFIG (dict): Model configuration.
        tokenizer: Tokenizer instance.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to run the model on (CPU or GPU).
        num_epochs (int): Number of training epochs.
        verbose (bool): Enable verbose output.
    """
    # Initialize the GPT model and move it to the specified device
    model = GPTModel(GPT_CONFIG)
    model.to(device)

    # Set up the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=GPT_CONFIG["learning_rate"],
        weight_decay=GPT_CONFIG["weight_decay"],
    )

    if verbose:
        print("Starting training...")

    # Train the model and capture training/validation losses
    train_losses, val_losses, tokens_seen = train_model_simple(
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        num_epochs=num_epochs,
        eval_freq=1,
        eval_iter=1,
        start_context="7 2 7",
    )

    # Save the trained model to disk
    torch.save(model.state_dict(), "models/path_finder.model")

    if verbose:
        print("Training completed. Model saved to 'models/path_finder.model'.")

    # Plot and save the training and validation loss curves
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)


def main():
    """
    Parse command-line arguments and run GPT training if specified.
    """
    parser = argparse.ArgumentParser(description="Train a GPT model and save it.")

    # Add command-line arguments
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training on the predefined data and save the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for detailed training prints.",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Specify the device to use for training (cpu or cuda). Default is cuda if available.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs. Default is 10.",
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
        "--config_file",
        type=str,
        default="config/gpt_config.json",
        help="Path to the GPT configuration file. Default is 'config/gpt_config.json'.",
    )

    # Parse command-line arguments
    args = parser.parse_args()

    # Run training if the --train flag is provided
    if args.train:
        GPT_CONFIG, tokenizer, train_loader, val_loader = setup_config_and_data(
            args.train_file, args.val_file, args.config_file
        )
        device = torch.device(args.device)
        train_and_save_model(
            GPT_CONFIG,
            tokenizer,
            train_loader,
            val_loader,
            device,
            num_epochs=args.epochs,
            verbose=args.verbose,
        )
    else:
        print("Please provide a valid option. Use --help for more details.")


if __name__ == "__main__":
    main()

# Example run:
# python run_gpt_train.py --train --epochs 10 --device cuda --config config/gpt_config.json --train_file data/train.txt --val_file data/val.txt --verbose

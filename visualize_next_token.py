import argparse
import matplotlib.pyplot as plt
import torch
import tiktoken

from utils_final import (
    GPTModel,
    generate_and_print_sample,
    load_config,
)


def main():
    """
    Visualize next token prediction given an input sequence using a pre-trained GPT model.
    """
    parser = argparse.ArgumentParser(
        description="Visualize next token prediction using a pre-trained GPT model."
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="config/gpt_config.json",
        help="Path to the model configuration file. Default is 'config/gpt_config.json'.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="models/gpt_model.pth",
        help="Path to the pre-trained model file. Default is 'models/gpt_model.pth'.",
    )
    parser.add_argument(
        "--input_text",
        type=str,
        default="19 94 19 27 34 46 58 65 67 82 87 94",
        help="Input text to analyze. Default is '19 94 19 27 34 46 58 65 67 82 87 94'.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="next_token_probabilities.png",
        help="Path to save the next token probability plot. Default is 'next_token_probabilities.png'.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run the model on. Options are 'cpu' or 'cuda'. Default is 'cpu'.",
    )
    parser.add_argument(
        "--proba_threshold",
        type=float,
        default=0.001,
        help="Threshold for plotting the token probabilities. Default is 0.001.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose mode for detailed output.",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    if args.verbose:
        print(f"Using device: {device}")

    # Load the model configuration
    GPT_CONFIG = load_config(args.config_file)

    # Initialize the model and load pre-trained weights
    model = GPTModel(GPT_CONFIG)
    model.load_state_dict(torch.load(args.model_file, map_location=device))

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.verbose:
        print(f"Model and tokenizer loaded successfully.")
        print(f"Input text: {args.input_text}")

    # Generate and visualize next token predictions
    path_hat = generate_and_print_sample(
        model,
        tokenizer,
        device,
        args.input_text,
        max_new_tokens=1,
        plot_best_tokens_prob=True,
        print_output=True,
        proba_threshold=args.proba_threshold,
        save_path=args.save_path,
    )

    if args.verbose:
        print(f"Next token prediction visualized and saved to {args.save_path}.")


if __name__ == "__main__":
    main()

# Example run:
# python visualize_next_token.py --config_file config/gpt_config.json --model_file models/gpt_model.pth --input_text "19 94 19 27 34 46 58 65 67 82 87 94" --save_path img/next_token_probabilities.png --verbose

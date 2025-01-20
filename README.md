# Graph Path Learning with Transformers

This repository contains a project to train Transformers on graph-based data, testing the model's ability to learn paths and perform path searches between source and target nodes. The project leverages a Transformer-based architecture inspired by Sebastian Raschka's book series.

## Overview

The project includes the following main functionalities:

1. **Data Generation:**
   - Generates random Directed Acyclic Graphs (DAGs).
   - Creates train and validation datasets with paths between nodes.

2. **Training:**
   - Trains a Transformer model on the generated data to predict paths within a graph.

3. **Validation:**
   - Validates the trained model's ability to predict paths, checking for accuracy and errors such as hallucinations or unreachable nodes.

---

## File Structure

- **`run_data_generation.py`**: Script to generate random DAGs and path datasets.
- **`run_gpt_train.py`**: Script to train the Transformer model.
- **`run_validation.py`**: Script to validate the trained model on the graph-based path prediction task.
- **`utils_final.py`**: Utility functions for model implementation, data handling, and visualization.

---

## How to Run the Project

### Prerequisites

Ensure you have Python 3.7+ installed and the required dependencies:

```bash
pip install -r requirements.txt
```

### Step 1: Data Generation

Generate a random DAG and create training/validation datasets:

```bash
python run_data_generation.py \
    --nodes 100 \
    --probability 0.1 \
    --ratio 0.8 \
    --train_file data/train_reasoning.txt \
    --val_file data/val_reasoning.txt \
    --dag_file data/DAG.gpickle \
    --seed 42 \
    --verbose
```

### Step 2: Model Training

Train the Transformer model on the generated dataset:

```bash
python run_gpt_train.py \
    --train \
    --epochs 10 \
    --device cuda \
    --config_file config/gpt_config.json \
    --train_file data/train_reasoning.txt \
    --val_file data/val_reasoning.txt \
    --verbose
```

### Step 3: Validation

Validate the trained model on the validation dataset:

```bash
python run_validation.py \
    --config_file config/gpt_config.json \
    --model_file models/path_finder.model \
    --train_file data/train_reasoning.txt \
    --val_file data/val_reasoning.txt \
    --dag_file data/DAG.gpickle \
    --verbose
```

---

## Example Output

Below is a sample terminal output during validation:

![Validation Example](example_output.png)

---

## Key Features

- Generates random graphs for diverse testing scenarios.
- Transformer-based architecture fine-tuned for graph traversal tasks.
- Detailed validation output to analyze model performance and error patterns.

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- Sebastian Raschka's books and resources for Transformer implementation.
- PyTorch for the deep learning framework.
- NetworkX for graph generation and manipulation.

Feel free to explore and contribute to this project!

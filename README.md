# Graph Path Learning with Transformers

This repository contains a project to train Transformers on graph-based data, testing the model's ability to learn paths and perform path searches between source and target nodes. The project leverages a Transformer-based architecture inspired by Sebastian Raschka's book series.

<img src="https://github.com/rmaestre/transformers_path_search/blob/main/img/run_example.png"></img>

## Overview

The project includes the following main functionalities:


### 1. **Python Debugger: Current File**
   - **Description**: Runs the currently open Python file in the integrated terminal.
   - **Execution**:
     ```bash
     python ${file}
     ```

---

### 2. **Python Debugger: Run Validation**
   - **Description**: Runs the validation script to evaluate the model's performance using specified configuration and data files.
   - **Execution**:
     ```bash
     python run_validation.py --config config/gpt_config_multi.json --model_file models/gpt_config_multi_model.pth --train_file data/train.txt --val_file data/val.txt --dag_file data/dag.gpickle --verbose
     ```

---

### 3. **Python Debugger: Visualize Attention**
   - **Description**: Visualizes attention weights from the Transformer model for a given input sequence.
   - **Execution**:
     ```bash
     python visualize_attention.py --config_file config/gpt_config_multi.json --model_file models/gpt_config_multi_model.pth --input_text "177 165 177 149 57 120 112 181 134 165" --head 0 1 2 --layer 0 1 --save_path img/attention_weights.png --verbose --use_power_scale
     ```

---

### 4. **Python Debugger: Visualize Next Token**
   - **Description**: Visualizes the next token probabilities predicted by the model for a given input sequence.
   - **Execution**:
     ```bash
     python visualize_next_token.py --config_file config/gpt_config_multi.json --model_file models/gpt_config_multi_model.pth --input_text "65 177 149 57 120" --save_path img/next_token_probabilities.png --verbose
     ```

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

## Key Features

- Generates random graphs for diverse testing scenarios.
- Transformer-based architecture fine-tuned for graph traversal tasks.
- Detailed validation output to analyze model performance and error patterns.

---


## Acknowledgments

- Sebastian Raschka's books and resources for Transformer implementation.
- Siwei Wang, Yifei Shen, Shi Feng, Haoran Sun, Shang-Hua Teng, Wei Chen. ALPINE: Unveiling the Planning Capability of Autoregressive Learning in Language .

Feel free to explore and contribute to this project!

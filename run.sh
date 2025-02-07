#!/bin/bash
cd src
python run_data_generation.py

python run_gpt_train.py --train --epochs 30 --config ../config/exp_1head_1layer.json --train_file ../data/train.txt --val_file ../data/val.txt --verbose --device cpu

python run_validation.py --config ../config/exp_1head_1layer.json --model_file ../models/exp_1head_1layer.pth --train_file ../data/train.txt --val_file ../data/val.txt --dag_file ../data/dag.gpickle --verbose

python visualize_attention.py --config_file ../config/exp_1head_1layer.json --model_file ../models/exp_1head_1layer.pth --input_text "187 109 187 141 175 160 161 39 59 12 103 159 46 171 109" --head 012  --verbose --use_power_scale
python visualize_next_token.py --config_file ../config/exp_1head_1layer.json --model_file ../models/exp_1head_1layer.pth --input_text "92 143 92 60 90" --save_path ../img/next_token_probabilities.png --verbose

chmod +x run.sh
#!/bin/bash

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 <n_functions_to_sample> <n_subset_to_choose> <max_instructions> <seed_data>"
    exit 1
fi

n_functions_to_sample=$1
n_subset_to_choose=$2
max_instructions=$3
seed_data=$4

DATASET="..." # (numpy/pandas/aws)
MODEL="..."
TOOLS_PATH="..."
SAVE_PATH="..."

IFS=',' read -ra n_subset_array <<< "$n_subset_to_choose"

python main.py \
    --iterations 10000000 \
    --dataset $DATASET \
    --max_instructions $max_instructions \
    --model $MODEL \
    --port 8000 \
    --tools_path $TOOLS_PATH \
    --max_concurrent_tasks 30 \
    --n_functions_to_sample $n_functions_to_sample \
    --n_subset_to_choose "${n_subset_array[@]}" \
    --n_in_context_examples 3 \
    --save_path $SAVE_PATH \
    --use_polisher \
    --seed_data $seed_data

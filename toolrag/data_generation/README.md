# Task-specific Tool Retrieval Dataset Generation

To generate your own task-specific tool-retrieval, follow these 3 steps:

### 1. Prepare your tools
Prepare a file containing your tools and descriptions with the following format
```
{
    "<your function name>": {
        "new_func": "<new name>", # We used this field in some ToolBank datasets to assign fake names during dataset generation but it is completely optional. You can just use <your function name>,
        "description": "<function description>"
    },
    ...
}
```

## 2. (Optional) Write 2-3 seed in-context examples (ICE)
You can optionally write some initial ICEs for your task-specific dataset. For this, you can create task-specific `InstructionGenerator` class that inherits from `BaseInstructionGenerator`and a `InstructionPolisher` class that inherits from `BaseInstructionPolisher`. You can check some examples of this in `/bank_aws`, `/bank_numpy`, and `/bank_pandas` folders. After creating the classes and the ICE according to the appropriate format, add these to `main.py`

## 3. Generate data
Use the `generate_data_script.sh` to generate synthetic data. You need to specify which model to use and appropriate file paths.
```
DATASET="..." # (numpy/pandas/aws)
MODEL="..."
TOOLS_PATH="..."
SAVE_PATH="..."

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
```

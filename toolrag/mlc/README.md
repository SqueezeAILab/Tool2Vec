# Multi-Label Classification

## For ToolBank Datasets
### Data preprocessing
```
python format_train_data.py \
    --all_tools_path <pandas_tools_path> \
    --train_data_path <pandas_train_path> \
    --output_path pandas.pt
```

### Model Training
```
python train.py \
    --train_data_path pandas.pt \
    --valid_data_path <pandas_valid_path> \
    --all_tools_path <pandas_tools_path> \
    --checkpoint_dir pandas_checkpoints \
    --model_name microsoft/deberta-v3-base \
    --epochs 100 \
    --wandb_name pandas \
    --num_labels 1655
```

### Model Evaluation
```
python test.py \
    --test_data_path  <pandas_test_path> \
    --all_tools_path <pandas_tools_path> \
    --model_name microsoft/deberta-v3-base \
    --model_path pandas_checkpoints/model_recall_at_3.pt \
    --num_labels 1655
```

## For ToolBench
### Model Training
```
python train_toolbench.py \
    --category G3 \
    --toolbench_data_dir <toolbench_data_dir> \
    --model_name microsoft/deberta-v3-base \
    --num_labels 1605 \
    --epochs 30 \
    --batch_size 32 \
    --lr 5e-5 \
    --wd 0.01 \
    --num_linear_warmup_steps 1600 \
    --wandb_name toolbench \
    --checkpoint_dir toolbench_checkpoints
```

### Model Evaluation
```
python test_toolbench.py \
    --model_name microsoft/deberta-v3-base \
    --model_path toolbench_checkpoints/model_recall_at_3.pt \
    --toolbench_data_dir <toolbench_data_dir> \
    --category G3
```
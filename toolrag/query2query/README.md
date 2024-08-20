# T2V Embedding Fine-Tuning

## For ToolBank Datasets
### Data preprocessing
```
python format_train_data.py \
    --model intfloat/e5-base-v2 \
    --train_data_path <numpy_train_path> \
    --output_path numpy.pt
```

### Model Training
```
python train.py \
    --train_data_path numpy.pt \
    --model_name intfloat/e5-base-v2 \
    --epochs 1 \
    --batch_size 32 \
    --lr 2e-5 \
    --wd 0.01 \
    --num_linear_warmup_steps 100 \
    --checkpoint_dir numpy_checkpoints \
    --wandb_name numpy
```

### Model Evaluation
Refer to the normal T2V evaluation code, but pass in the checkpoint of the trained model.

## For ToolBench
### Data preprocessing
```
python format_train_data_toolbench.py \
    --model intfloat/e5-base-v2 \
    --toolbench_data_dir <toolbench_data_dir> \
    --category G3 \
    --output_path toolbench.pt
```

### Model Training
```
python train_toolbench.py \
    --train_data_path toolbench.pt \
    --model_name intfloat/e5-base-v2 \
    --epochs 1 \
    --batch_size 32 \
    --lr 1e-5 \
    --wd 0.01 \
    --num_linear_warmup_steps 1600 \
    --checkpoint_dir toolbench_checkpoints \
    --wandb_name toolbench
```

### Model Evaluation
Refer to the normal T2V evaluation code, but pass in the checkpoint of the trained model.

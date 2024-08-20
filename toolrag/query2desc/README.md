# Description-Based Retrieval

## For ToolBank Datasets
### Data preprocessing
```
python format_train_data.py \
    --model intfloat/e5-base-v2 \
    --train_data_path <numpy_train_path> \
    --output_path numpy.pt \
    --tool_descriptions_path <numpy_descriptions_path>
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
    --seed 0 \
    --margin 1 \
    --checkpoint_dir numpy_checkpoints \
    --wandb_name numpy
```

### Model Evaluation
```
python test.py \
    --test_data_path <numpy_test_path> \
    --all_tools_path <numpy_tools_path>  \
    --tool_descriptions_path <numpy_descriptions_path> \
    --model intfloat/e5-base-v2 \
    --checkpoint numpy_checkpoints/model_step_5000.pt
```

## For ToolBench
We use the ToolBench retriever trained in the ToolBench paper. The model can be found [here](https://huggingface.co/ToolBench/ToolBench_IR_bert_based_uncased).
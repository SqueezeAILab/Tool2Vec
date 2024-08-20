# ToolRefiner

## Model Training
```
MODEL="microsoft/deberta-v3-xsmall"
LR=8e-4
STD=0.2
WD=0.01
BATCH_SIZE=32
EPOCH=30
TOOL=64
WR=100

TRAIN_DATA_PATH="...json"
VAL_DATA_PATH="...json"
EMBEDDING_DATA_PATH="...pkl"
TOOL_NAME_PATH="...json"
TRAIN_TOP_K_DIR="...json"
VAL_TOP_K_DIR="...json"
CHECKPOINT_DIR="..."
WANDB_NAME="..."

python train_query_nt.py \
    --model $MODEL \
    --lr $LR \
    --use_amp \
    --std $STD \
    --wd $WD \
    --num_tools_to_be_presented $TOOL \
    --num_linear_warmup_steps $WR \
    --batch_size $BATCH_SIZE \
    --num_epochs $EPOCH \
    --training_data_dir ${TRAIN_DATA_PATH} \
    --test_data_dir ${VAL_DATA_PATH} \
    --tool_embedding_dir ${EMBEDDING_DATA_PATH} \
    --tool_name_dir ${TOOL_NAME_PATH} \
    --train_tool_top_k_retrieval_dir ${TRAIN_TOP_K_DIR} \
    --valid_tool_top_k_retrieval_dir ${VAL_TOP_K_DIR} \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --wandb_name $WANDB_NAME \
```

## Model Evaluation
```
BATCH_SIZE=64
TOOL=64

CHECKPOINT_DIR="..."
CHECKPOINT_EPOCH="..."
TEST_DIR="...json"
EMBEDDING_DATA_PATH="...pkl"
TOOL_NAME_PATH="...json"
TEST_TOP_K_DIR="...json"

python evaluate_query_nt.py \
    --checkpoint_dir ${CHECKPOINT_DIR} \
    --checkpoint_epoch $CHECKPOINT_EPOCH \
    --batch_size $BATCH_SIZE
     --test_data_dir ${VAL_DATA_PATH} \
    --tool_embedding_dir ${EMBEDDING_DATA_PATH} \
    --tool_name_dir ${TOOL_NAME_PATH} \
    --valid_tool_top_k_retrieval_dir ${VAL_TOP_K_DIR} \
    --num_tools_to_be_presented $TOOL
```

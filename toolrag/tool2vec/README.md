# Tool2Vec

## Generate Tool2Vec Embeddings
For generating tool2vec embeddings, you need to provide the path to the dataset, output path, output filename and model. The available models are: `text-embedding-3-small`, `e5-small`, `e5-base`, `e5-large`, and `mxbai-large`.
To use OpenAI model, you need to set the `AZURE_ENDPOINT`, `AZURE_OPENAI_API_KEY`, and `AZURE_OPENAI_API_VERSION` environment variables.
For other models, you need to provide the path to the model checkpoint.

For example, to generate embeddings using the `text-embedding-3-small` model with the dataset located at `...json` and save the embeddings to `...pkl`, you can run the following command:
```
AZURE_ENDPOINT="..."
AZURE_OPENAI_API_KEY="..."
AZURE_OPENAI_API_VERSION="..."
DATA_PATH="...json"
OUTPUT_PATH="..."
OUTPUT_FILE_NAME="....pkl"

python embedding_generator.py \
    --data_path ${DATA_PATH} \
    --output_path "..." \
    --output_file_name "..." \
    --model "azure"
```

To generate embeddings using the `e5-small` model with the dataset located at `...json` and save the embeddings to `...pkl`, you can run the following command:
```
DATA_PATH="...json"
OUTPUT_PATH="..."
OUTPUT_FILE_NAME="....pkl"

python embedding_generator.py \
    --data_path ${DATA_PATH} \
    --output_path "..." \
    --output_file_name "..." \
    --model "e5-small" \
    --checkpoint_path "..." \
    --use_checkpoint
```

## Model Evaluation
```
VAL_DATA_PATH="...json"
T2V_EMBEDDING_DATA_PATH="...pkl"
OUTPUT_FILE_NAME="....txt"

python evaluate_t2v_embedding.py \
    --valid_data_path ${VAL_DATA_PATH} \
    --tool_embedding_dir ${T2V_EMBEDDING_DATA_PATH} \
    --output_file_name ${OUTPUT_FILE_NAME}
```

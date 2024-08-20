"""
Evaluate the tool2vec embedding using recall@k metric.
Usage:
    python tool2vec/evaluate_t2v_embedding.py
"""

import json
import argparse
import numpy as np
import pickle

from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--valid_data_path",
    type=str,
    help="Path to the file containing valid/test data",
    required=True,
)
argparser.add_argument(
    "--t2v_embedding_path",
    type=str,
    help="Path to tool2vec embedding",
    required=True,
)
argparser.add_argument(
    "--top_k_list",
    type=int,
    nargs="+",
    default=[3, 5, 7, 10, 12, 32, 64, 128, 256],
    help="Top k list, default is [3, 5, 7, 10, 12, 32, 64, 128, 256]",
)
argparser.add_argument("--debug", action="store_true", help="Debug mode")
argparser.add_argument("--output_file_name", type=str, help="Output file")

args = argparser.parse_args()


def compute_recall_k(
    valid_data: list,
    t2v_data: np.array,
    tool_to_idx: dict[str, int],
    top_k: int = 5,
) -> float:
    """
    Compute recall@k for the given data and t2v embeddings.

    Args:
        valid_data (list): The valid/test data.
        t2v_data (np.array): The t2v embeddings.
        tool_to_idx (dict): The tool to index mapping.
        top_k (int): The top k value.

    Returns:
        float: The recall@k value.
    """
    total_recall = 0
    total_ndcg = 0

    for _, valid in enumerate(valid_data):
        valid_tool_embedding = np.array(valid["function_embedding"])

        # calculate cosine similarity
        valid_tool_embedding = valid_tool_embedding.reshape(1, -1)
        cosine_sim = np.dot(t2v_data, valid_tool_embedding.T)

        # get top k
        top_k_idx = np.argsort(cosine_sim.flatten())[::-1][:top_k]

        correct_tool_idx = set(
            [tool_to_idx.get(tool, -1) for tool in valid["functions"]]
        )

        # calculate recall
        recall = len(set(top_k_idx) & correct_tool_idx) / len(correct_tool_idx)
        total_recall += recall

        # calculate ndcg
        dcg = 0
        for idx, tool in enumerate(top_k_idx):
            if tool in correct_tool_idx:
                dcg += 1 / np.log2(2 + idx)

        max_dcg = 0
        for idx in range(min(top_k, len(correct_tool_idx))):
            max_dcg += 1 / np.log2(2 + idx)
        ndcg = dcg / max_dcg
        total_ndcg += ndcg

    return total_recall / len(valid_data), total_ndcg / len(valid_data)


valid_data_path = Path(args.valid_data_path)
t2v_embedding_path = Path(args.t2v_embedding_path)

print("Loading data...")
with open(valid_data_path, "r") as f:
    valid_data = json.load(f)

with open(t2v_embedding_path, "rb") as f:
    t2v_embedding = pickle.load(f)

print("Data loaded")
print("Valid/test data length:", len(valid_data))
print("The number t2v embeddings:", len(t2v_embedding))

# idx to key
idx_to_tool, tool_to_idx = {}, {}
for idx, tool in enumerate(t2v_embedding):
    idx_to_tool[idx] = tool

    if tool not in tool_to_idx:
        tool_to_idx[tool] = idx

t2v_data_np = np.array([t2v_embedding[idx_to_tool[idx]] for idx in idx_to_tool])

recalls_at_k = {}
for k in args.top_k_list:
    recall_k, ndcg_k = compute_recall_k(
        valid_data=valid_data,
        t2v_data=t2v_data_np,
        tool_to_idx=tool_to_idx,
        top_k=k,
    )
    print(f"Recall@{k}: {recall_k}")
    print(f"NDCG@{k}: {ndcg_k}")
    recalls_at_k[k] = recall_k

if args.output_file_name:
    with open(args.output_file_name, "w") as f:
        for k, recall in recalls_at_k.items():
            f.write(f"Recall@{k}: {recall}\n")

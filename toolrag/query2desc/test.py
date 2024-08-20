import argparse
import json
import torch

import numpy as np
from toolrag.models.e5 import E5Model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--checkpoint", type=str)
parser.add_argument("--test_data_path", type=str, required=True)
parser.add_argument("--all_tools_path", type=str, required=True)
parser.add_argument("--tool_descriptions_path", type=str, required=True)
args = parser.parse_args()

model = E5Model(args.model)
if args.checkpoint:
    model.load_checkpoint(args.checkpoint)

with open(args.all_tools_path, "r") as f:
    all_tools = json.load(f)

with open(args.tool_descriptions_path, "r") as f:
    tool_descriptions = json.load(f)

tool_descriptions_tmp = []
unique_tools = list(set(all_tools.values()))
unique_tools.sort()

for tool in unique_tools:
    for k, v in tool_descriptions.items():
        if v["new_func"] == tool:
            tool_descriptions_tmp.append(v["description"])
            break

tool_descriptions = tool_descriptions_tmp
batch_size = 32
tool_embeddings = []
for i in range(0, len(tool_descriptions), batch_size):
    tool_embeddings.extend(model.embed_docs(tool_descriptions[i : i + batch_size]).detach().cpu())

tool_embeddings = torch.vstack(tool_embeddings)

with open(args.test_data_path, "r") as f:
    test_data = json.load(f)

all_predicted_tools = []
for data in test_data:
    if data["refined_instruction"]:
        instruction = data["refined_instruction"]
    else:
        instruction = data["instruction"]

    inst_embedding = model.embed_queries([instruction]).detach().cpu()
    top_indices = model.get_top_k_docs(inst_embedding, tool_embeddings, 256)
    predicted_tools = [unique_tools[i] for i in top_indices]
    all_predicted_tools.append(predicted_tools)

for k in [3, 5, 7, 10, 12]:
    recalls = []
    for idx, data in enumerate(test_data):
        predicted_tools = all_predicted_tools[idx][:k]
        gt_tools = data["functions"]
        recall = len(set(predicted_tools).intersection(gt_tools)) / len(gt_tools)
        recalls.append(recall)

    print(f"Recall@{k}: {np.mean(recalls)}")

import argparse
import json

import torch
from toolrag.models.e5 import E5Model
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--tool_descriptions_path", type=str, required=True)
parser.add_argument("--train_data_path", type=str, required=True)
args = parser.parse_args()

model = E5Model(args.model)

with open(args.tool_descriptions_path, "r") as f:
    tool_descriptions_dict = json.load(f)

tool_descriptions = [
    tool_descriptions_dict[tool]["description"]
    for tool in tool_descriptions_dict.keys()
]
tools = list(
    tool_descriptions_dict[tool]["new_func"] for tool in tool_descriptions_dict.keys()
)

tool_embeddings = []
batch_size = 32
tool_embeddings = []
for i in range(0, len(tool_descriptions), batch_size):
    batch = tool_descriptions[i : i + batch_size]
    batch_embeddings = model.embed_docs(batch).detach().cpu()
    tool_embeddings.extend(batch_embeddings)

tool_embeddings = torch.stack(tool_embeddings)

with open(args.train_data_path, "r") as f:
    train_data = json.load(f)

all_triplets = []
count = 0
for data in train_data:
    count += 1
    if data["refined_instruction"]:
        instruction = data["refined_instruction"]
    else:
        instruction = data["instruction"]

    # embed instruction
    inst_embedding = model.embed_queries([instruction]).detach().cpu()

    # find closest tools
    top_tools = [
        tools[i] for i in model.get_top_k_docs(inst_embedding, tool_embeddings, 10)
    ]
    true_tools = data["functions"]

    # Get random negatives
    hard_negatives = []
    random_tools = np.random.choice(tools, 5, replace=False)
    for tool in random_tools:
        if tool not in true_tools:
            hard_negatives.append(f"passage: {tool_descriptions[tools.index(tool)]}")

    # get positives
    positives = [
        f"passage: {tool_descriptions[tools.index(tool)]}" for tool in true_tools
    ]

    # get triplets
    triplets = [
        (instruction, positive, negative)
        for positive in positives
        for negative in hard_negatives
    ]
    all_triplets.extend(triplets)


# Create dataset for triplets
class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


# save dataset
dataset = TripletDataset(all_triplets)
torch.save(dataset, args.output_path)

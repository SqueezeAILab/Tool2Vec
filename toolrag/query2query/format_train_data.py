import argparse
import json

import torch

from toolrag.models.e5 import E5Model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--train_data_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
args = parser.parse_args()

model = E5Model(args.model)

with open(args.train_data_path, "r") as f:
    train_data = json.load(f)

instructions = []
for data in train_data:
    if data.get("refined_instruction"):
        instructions.append(data.get("refined_instruction"))
    else:
        instructions.append(data["instruction"])

instruction_embeddings = torch.vstack(
    [
        model.embed_queries(instructions[i : i + 64]).detach().cpu()
        for i in range(0, len(instructions), 64)
    ]
)
all_triplets = []
for data in train_data:
    if data.get("refined_instruction"):
        instruction = data.get("refined_instruction")
    else:
        instruction = data["instruction"]

    inst_embedding = model.embed_queries([instruction]).detach().cpu()

    top_instruction_idxs = model.get_top_k_docs(
        inst_embedding, instruction_embeddings, 100
    )
    true_tools = set(data["functions"])

    hard_negatives = []
    positives = []
    for idx in top_instruction_idxs:
        if set(train_data[idx]["functions"]) & true_tools == set():
            hard_negatives.append(instructions[idx])
        else:
            positives.append(instructions[idx])
    positives = positives[:5]
    hard_negatives = hard_negatives[:5]

    triplets = [
        (instruction, positive, negative)
        for positive in positives
        for negative in hard_negatives
    ]
    all_triplets.extend(triplets)


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


dataset = TripletDataset(all_triplets)
torch.save(dataset, args.output_path)

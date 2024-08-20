import argparse
import os

import pandas as pd
import torch

from toolrag.models.e5 import E5Model

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="intfloat/e5-base-v2")
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--category", type=str, required=True)
parser.add_argument("--toolbench_data_dir", type=str, required=True)
args = parser.parse_args()

TOOLBENCH_DATA_DIR = args.toolbench_data_dir

model = E5Model(args.model)

category_dir = os.path.join(TOOLBENCH_DATA_DIR, args.category)

queries_path = os.path.join(category_dir, "train.query.txt")
qrels_path = os.path.join(category_dir, "qrels.train.tsv")

queries_df = pd.read_csv(queries_path, sep="\t", header=None, names=["qid", "query"])
qrels_df = pd.read_csv(
    qrels_path, sep="\t", header=None, names=["qid", "useless", "docid", "label"]
)

queries = queries_df["query"].tolist()

query_embeddings = torch.vstack(
    [
        model.embed_queries(queries[i : i + 64]).detach().cpu()
        for i in range(0, len(queries), 64)
    ]
)

all_triplets = []
for row in queries_df.itertuples():
    qid = row.qid
    query = row.query

    query_embedding = model.embed_queries([query]).detach().cpu()
    query_docids = set(qrels_df[qrels_df.qid == qid]["docid"].tolist())

    most_similar_query_idxs = model.get_top_k_docs(
        query_embedding, query_embeddings, 20
    )
    hard_negatives = []
    positives = []
    for idx in most_similar_query_idxs:
        similar_row = queries_df.iloc[idx]
        similiar_query_docids = set(
            qrels_df[qrels_df.qid == similar_row.qid]["docid"].tolist()
        )
        if len(query_docids & similiar_query_docids) == 0:
            hard_negatives.append(similar_row.query)
        else:
            positives.append(similar_row.query)

    hard_negatives = hard_negatives[:5]
    positives = positives[:5]

    triplets = [
        (query, positive, negative)
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

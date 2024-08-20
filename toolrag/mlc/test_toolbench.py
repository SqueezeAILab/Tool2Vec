import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

CATEGORIES_TO_LABELS = {"G1": 10439, "G2": 13142, "G3": 1605}


def test(args) -> None:
    # Load the tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=CATEGORIES_TO_LABELS[args.category]
    )

    # Move the model to the GPU if available
    device = torch.device("cuda:0")
    model.to(device)

    # Load checkpoint
    checkpoint_path = args.model_path
    checkpoint = torch.load(checkpoint_path)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set the model to evaluation mode
    model.eval()

    # Refer to the following link for more information on the code implementation:
    # https://github.com/OpenBMB/ToolBench/blob/master/toolbench/retrieval/train.py#L79
    test_queries_df = pd.read_csv(
        os.path.join(TOOLBENCH_DATA_DIR, "test.query.txt"),
        sep="\t",
        names=["qid", "query"],
    )
    labels_df = pd.read_csv(
        os.path.join(TOOLBENCH_DATA_DIR, "qrels.test.tsv"),
        sep="\t",
        names=["qid", "useless", "docid", "label"],
    )
    ir_relevant_docs = {}
    for row in labels_df.itertuples():
        ir_relevant_docs.setdefault(row.qid, set()).add(row.docid)

    qid_to_labels = defaultdict(list)
    for _, row in labels_df.iterrows():
        assert row["useless"] == 0 and row["label"] == 1, row
        qid_to_labels[row["qid"]].append(row["docid"])

    def predict(texts, labels):
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            logits = outputs.logits
            predictions = torch.sigmoid(logits).cpu().numpy()

        return predictions, outputs.loss.item()

    predicted_tools = {}
    valid_loss = 0
    for row in test_queries_df.itertuples():
        qid = row.qid
        query = row.query

        labels = [0] * CATEGORIES_TO_LABELS[args.category]
        for tool_id in qid_to_labels[qid]:
            labels[tool_id - 1] = 1
        labels = torch.LongTensor(labels).unsqueeze(0).to(device)
        probs, loss = predict([query], labels)
        probs = probs[0]
        valid_loss += loss

        predicted_tools[(qid, query)] = [1 + x for x in np.argsort(probs)[::-1]]

    # compute recalls
    for k in [3, 5, 7, 10, 12]:
        recalls = []
        ndcgs = []
        for (qid, query), predicted_tools_cur in predicted_tools.items():
            gt_tools = ir_relevant_docs[qid]
            predicted_tools_i = predicted_tools_cur[:k]
            recall = len(set(predicted_tools_i).intersection(gt_tools)) / len(gt_tools)

            dcg = 0
            for idx, tool in enumerate(predicted_tools_i):
                if tool in gt_tools:
                    dcg += 1 / np.log2(2 + idx)

            max_dcg = 0
            for idx in range(min(k, len(gt_tools))):
                max_dcg += 1 / np.log2(2 + idx)
            ndcg = dcg / max_dcg

            ndcgs.append(ndcg)
            recalls.append(recall)
        print(f"Recall@{k}: {np.mean(recalls) * 100}")
        print(f"NDCG@{k}: {np.mean(ndcgs) * 100}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        choices=["G1", "G2", "G3"],
        help="ToolBench category",
        required=True,
    )
    parser.add_argument(
        "--model_name", type=str, help="Model name to use for testing", required=True
    )
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the model checkpoint",
        required=True,
    )
    parser.add_argument(
        "--toolbench_data_dir",
        type=str,
        help="Path to the ToolBench data directory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    TOOLBENCH_DATA_DIR = os.path.join(args.toolbench_data_dir, args.category)
    test(args)

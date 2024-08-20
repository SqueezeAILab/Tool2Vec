import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb
from toolrag.tool_reranker.utils import set_seed


class ToolBenchMLCDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train_queries_df: pd.DataFrame,
        qid_to_labels: dict[list],
        total_tools: int,
    ) -> None:
        """
        Args:
            train_queries_df (pd.DataFrame): DataFrame containing training queries.
            qid_to_labels (dict[list]): Dictionary mapping query IDs to tool labels.
            total_tools (int): Total number of tools.
        """
        self.train_queries_df = train_queries_df
        self.qid_to_labels = qid_to_labels
        self.total_tools = total_tools

    def __len__(self) -> int:
        return len(self.train_queries_df)

    def __getitem__(self, idx: int) -> tuple[str, torch.LongTensor]:
        qid = self.train_queries_df.iloc[idx]["qid"]
        query = self.train_queries_df.iloc[idx]["query"]
        labels = [0] * self.total_tools
        for tool_id in self.qid_to_labels[qid]:
            # Need to subtract 1 here because the tool ids are 1-indexed
            labels[tool_id - 1] = 1
        return query, torch.LongTensor(labels)


def validate(args, model, tokenizer, device, current_step, epoch) -> None:
    with torch.no_grad():
        category_dir = os.path.join(TOOLBENCH_DATA_DIR, args.category)
        # Refer to the following link for more information on the code implementation:
        # https://github.com/OpenBMB/ToolBench/blob/master/toolbench/retrieval/train.py#L79
        test_queries_df = pd.read_csv(
            os.path.join(category_dir, "test.query.txt"),
            sep="\t",
            names=["qid", "query"],
        )
        labels_df = pd.read_csv(
            os.path.join(category_dir, "qrels.test.tsv"),
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

        # Function to perform inference
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

            labels = [0] * args.num_labels
            for tool_id in qid_to_labels[qid]:
                labels[tool_id - 1] = 1
            labels = torch.LongTensor(labels).unsqueeze(0).to(device)

            probs, loss = predict([query], labels)
            probs = probs[0]
            valid_loss += loss

            predicted_tools[(qid, query)] = [1 + x for x in np.argsort(probs)[::-1]]

        wandb.log(
            {"valid/loss": valid_loss / len(test_queries_df)},
            step=current_step,
        )

        # compute recalls
        recalls_at_k = {}
        for k in [3, 5, 7, 10, 12]:
            recalls = []
            for (qid, query), predicted_tools_cur in predicted_tools.items():
                gt_tools = ir_relevant_docs[qid]
                predicted_tools_i = predicted_tools_cur[:k]
                recall = len(set(predicted_tools_i).intersection(gt_tools)) / len(
                    gt_tools
                )
                recalls.append(recall)

            wandb.log(
                {f"valid/recall@{k}": np.mean(recalls), "epoch": epoch},
                step=current_step,
            )
            recalls_at_k[k] = np.mean(recalls)

        return recalls_at_k


def train(args: argparse.Namespace) -> None:
    """
    Trains a sequence classification model.

    Args:
        args (argparse.Namespace): Command-line arguments.

    Returns:
        None
    """
    # Load the tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=args.num_labels,
    )

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load training data
    category_dir = os.path.join(TOOLBENCH_DATA_DIR, args.category)
    documents_df = pd.read_csv(os.path.join(category_dir, "corpus.tsv"), sep="\t")
    doc_ids = set(documents_df["docid"].tolist())

    # Refer to the following link for more information on the code implementation:
    # https://github.com/OpenBMB/ToolBench/blob/master/toolbench/retrieval/train.py#L79
    train_queries_df = pd.read_csv(
        os.path.join(category_dir, "train.query.txt"),
        sep="\t",
        names=["qid", "query"],
    )
    labels_df = pd.read_csv(
        os.path.join(category_dir, "qrels.train.tsv"),
        sep="\t",
        names=["qid", "useless", "docid", "label"],
    )
    qid_to_labels = defaultdict(list)

    for _, row in labels_df.iterrows():
        assert row["useless"] == 0 and row["label"] == 1, row
        qid_to_labels[row["qid"]].append(row["docid"])

    total_tools = len(doc_ids)
    assert total_tools == args.num_labels
    dataset = ToolBenchMLCDataset(train_queries_df, qid_to_labels, total_tools)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(dataloader) * args.epochs
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    if args.checkpoint_dir:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    current_step = 0
    best_recall_at_3_so_far = -1
    best_recall_at_5_so_far = -1
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        model.train()

        for batch in tqdm(dataloader, desc="Batches", leave=False):
            b_instructions, b_labels = batch
            b_input = tokenizer(
                b_instructions,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            b_input = b_input.to(device)
            b_labels = b_labels.to(device)

            # Forward pass
            outputs = model(**b_input, labels=b_labels)
            loss = outputs.loss

            # Update learning rate
            if current_step < args.num_linear_warmup_steps:
                new_lr = args.lr * current_step / args.num_linear_warmup_steps
                for param_group in optimizer.param_groups:
                    param_group["lr"] = new_lr
            else:
                lr_scheduler.step()

            # Take gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log the loss
            if current_step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=current_step,
                )
            current_step += 1

        # Get validation performance
        recalls_at_k = validate(args, model, tokenizer, device, current_step, epoch + 1)

        # Save model checkpoint
        if args.checkpoint_dir:
            if recalls_at_k[3] > best_recall_at_3_so_far:
                best_recall_at_3_so_far = recalls_at_k[3]
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, "model_recall_at_3.pt"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    },
                    checkpoint_path,
                )

            if recalls_at_k[5] > best_recall_at_5_so_far:
                best_recall_at_5_so_far = recalls_at_k[5]
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, "model_recall_at_5.pt"
                )
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "lr_scheduler_state_dict": lr_scheduler.state_dict(),
                    },
                    checkpoint_path,
                )


def parse_args() -> argparse.Namespace:
    """
    Parses the command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        choices=["G1", "G2", "G3"],
        help="ToolBench category",
        required=True,
    )
    parser.add_argument(
        "--model_name", type=str, help="Model name to use for training", required=True
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--num_labels", type=int, help="Number of labels", default=96)
    parser.add_argument("--wandb_name", type=str, default=None, help="Name for wandb")

    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of epochs for training"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    parser.add_argument(
        "--num_linear_warmup_steps",
        type=int,
        default=100,
        help="number of linear warmup steps",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--toolbench_data_dir",
        type=str,
        help="Path to the ToolBench data directory",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    TOOLBENCH_DATA_DIR = args.toolbench_data_dir
    set_seed(args.seed)
    if args.wandb_name:
        wandb.init(project="t2v", config=args, name=args.wandb_name)
    train(args)

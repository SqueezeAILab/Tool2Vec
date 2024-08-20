import argparse
import json
import os

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import wandb
from toolrag.mlc.format_train_data import MLCDataset
from toolrag.tool_reranker.utils import set_seed


def validate(args, model, tokenizer, device, current_step, epoch) -> None:
    with torch.no_grad():
        valid_data_path = args.valid_data_path
        with open(valid_data_path, "r") as f:
            valid_data = json.load(f)

        all_tools_path = args.all_tools_path
        with open(all_tools_path, "r") as f:
            all_tools = json.load(f)

        unique_tools = list(set(all_tools.values()))
        unique_tools.sort()

        def predict(texts):
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
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.sigmoid(logits).cpu().numpy()

            return predictions

        recalls_at_k = {}
        for k in [3, 5, 7, 10, 12]:
            recalls = []
            for data in valid_data:
                if data.get("refined_instruction"):
                    instruction = data.get("refined_instruction")
                else:
                    instruction = data["instruction"]

                # Get top k tools by probability
                probs = predict([instruction])[0]
                assert len(probs) == len(unique_tools)
                sorted_tools = sorted(
                    zip(unique_tools, probs), key=lambda x: x[1], reverse=True
                )
                predicted_tools = [tool for tool, _ in sorted_tools[:k]]

                gt_tools = data["functions"]

                # calculate recall
                recall = len(set(predicted_tools).intersection(gt_tools)) / len(
                    gt_tools
                )
                recalls.append(recall)

            recalls_at_k[k] = np.mean(recalls)
            wandb.log(
                {f"valid/recall@{k}": recalls_at_k[k], "epoch": epoch},
                step=current_step,
            )

        return recalls_at_k


def train(args: argparse.Namespace) -> None:
    # Load the tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=args.num_labels
    )

    # Move the model to the GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # load mlc_training.pt
    dataset: MLCDataset = torch.load(args.train_data_path)
    dataloader = DataLoader(dataset, batch_size=64)

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
        if (epoch + 1) % 10 == 0:
            recalls_at_k = validate(
                args, model, tokenizer, device, current_step, epoch + 1
            )

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
    parser = argparse.ArgumentParser()
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
        "--train_data_path", type=str, help="Path to the training data", required=True
    )
    parser.add_argument(
        "--valid_data_path", type=str, help="Path to the validation data"
    )
    parser.add_argument(
        "--all_tools_path",
        type=str,
        help="Path to the file containing all tools data",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if args.wandb_name:
        wandb.init(project="t2v", config=args, name=args.wandb_name)
    train(args)

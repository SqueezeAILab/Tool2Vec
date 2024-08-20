"""
python train_1q1t.py
"""

import gc
import os
import argparse
from collections import defaultdict

import numpy as np
import re
import torch
import torch.nn as nn
from toolrag.tool_reranker.evaluations import top_k_recall
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, DebertaV2Model

from toolrag.tool_reranker.t2v_datasets import T2VDatasetQueryNT, t2v_collator_query_nt
from toolrag.tool_reranker.utils import set_seed


class T2VPretrainedReranker(nn.Module):
    EMB_DIM_SIZE = {
        "microsoft/deberta-v3-xsmall": 384,
        "microsoft/deberta-v3-base": 768,
        "microsoft/deberta-v3-large": 1024,
    }

    def __init__(
        self, model_name, std=0.2, num_layer_to_freeze=0, use_cls=False, use_sep=False
    ):
        super().__init__()
        self.use_cls = use_cls
        self.use_sep = use_sep

        # The number of layers to freeze
        self.num_layer_to_freeze = num_layer_to_freeze

        # The device to store the model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # The linear layer to project the input embeddings to the BERT input dimension
        # We intialize the weights so that the second order stats of BERT embeddings are preserved
        # NOTE: This is only used for the t2v embedding
        model_emb_size = self.EMB_DIM_SIZE[model_name]
        self.embedding_projection: nn.Linear = nn.Linear(768, model_emb_size)
        nn.init.normal_(self.embedding_projection.weight, mean=0, std=std)

        deberta = DebertaV2Model.from_pretrained(model_name)
        self.encoder = deberta.encoder

        # Projection for the CLS token to get classifcation output
        self.linear = nn.Linear(model_emb_size, 1)

        # The embedding layer for the input tokens
        self.embedding = deberta.embeddings

        # Store the CLS and SEP token embeddings to be inserted later
        # NOTE: Just store the token ids for now, will need to add this at the
        # forward() method and pass it do the embedding layer.
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        cls_token_id: int = tokenizer.cls_token_id
        sep_token_id: int = tokenizer.sep_token_id

        self.register_buffer("cls_token_id", torch.tensor(cls_token_id))
        self.register_buffer("sep_token_id", torch.tensor(sep_token_id))

    def freeze_layers(self):
        for layer in self.encoder.layer[: self.num_layer_to_freeze]:
            for param in layer.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        tool_embedding: torch.Tensor,
    ):
        # input_ids: [batch_size, num_tokens]
        # tool_embedding: [batch_size, num_tools, 1536]
        input_ids = input_ids.to(self.device)
        tool_embedding = tool_embedding.to(self.device)

        # tool_embedding_proj: [batch_size, num_tools, 384]
        tool_embedding_proj = self.embedding_projection(tool_embedding)

        if self.use_cls:
            # Insert the cls token id at the beginning of the input_ids
            cls_token_id = self.cls_token_id.unsqueeze(0).expand(input_ids.shape[0], 1)
            input_ids = torch.cat([cls_token_id, input_ids], dim=1)
            attention_mask = torch.cat(
                [
                    torch.ones(
                        attention_mask.shape[0],
                        1,
                    ).to(self.device),
                    attention_mask,
                ],
                dim=1,
            )

        if self.use_sep:
            # Insert sep token at the last position of the query
            sep_token_id = self.sep_token_id.unsqueeze(0).expand(input_ids.shape[0], 1)
            input_ids = torch.cat([input_ids, sep_token_id], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(attention_mask.shape[0], 1).to(self.device),
                ],
                dim=1,
            )

        # input_embeddings: [batch_size, num_tokens, 384]
        input_embeddings = self.embedding(
            input_ids=input_ids,
        )

        num_tokens = input_embeddings.shape[1]

        # Concatenate the tool embedding with the input embeddings
        # embeddings: [batch_size, num_tokens+1, 384]
        embeddings = torch.cat([input_embeddings, tool_embedding_proj], dim=1)

        # attention_mask:
        # [CLS] [input_ids] [padding] [SEP] [tool_embeddings]
        # [1 1...1 0...0 1 1 ... 1] where 0...0 is padding
        num_tools = tool_embedding_proj.shape[1]
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(attention_mask.shape[0], num_tools).to(self.device),
            ],
            dim=1,
        )

        # encoder_output: [batch_size, num_tools, 384]
        encoder_output = self.encoder(embeddings, attention_mask).last_hidden_state[
            :, num_tokens:, :
        ]

        # out: [batch_size, num_tools]
        out = self.linear(encoder_output).squeeze(2)
        return out


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model", type=str, default="microsoft/deberta-v3-xsmall", help="model name"
    )
    argparser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    argparser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    argparser.add_argument(
        "--num_epochs", type=int, default=20, help="number of epochs"
    )
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument(
        "--num_linear_warmup_steps",
        type=float,
        default=100,
        help="number of linear warmup steps",
    )
    argparser.add_argument("--batch_size", type=int, default=256, help="batch size")

    argparser.add_argument(
        "--training_data_dir",
        type=str,
        required=True,
        help="training data directory",
    )
    argparser.add_argument(
        "--test_data_dir",
        type=str,
        required=True,
        help="test data directory",
    )
    argparser.add_argument(
        "--tool_embedding_dir",
        type=str,
        required=True,
        help="tool embedding directory",
    )
    argparser.add_argument(
        "--tool_name_dir",
        type=str,
        required=True,
        help="tool name directory",
    )
    argparser.add_argument(
        "--train_tool_top_k_retrieval_dir",
        type=str,
        default="",
        help="tool top k retrieval directory",
    )
    argparser.add_argument(
        "--valid_tool_top_k_retrieval_dir",
        type=str,
        default="",
        help="tool top k retrieval directory",
    )
    argparser.add_argument(
        "--num_layer_to_freeze",
        type=int,
        default=0,
        help="number of layers to freeze",
    )
    argparser.add_argument("--use_amp", action="store_true", help="use amp")
    argparser.add_argument(
        "--iters_to_accumulate",
        type=int,
        default=1,
        help="iters to accumulate gradients",
    )
    argparser.add_argument(
        "--num_tools_to_be_presented",
        type=int,
        default=64,
        help="number of tools to bepresented",
    )

    argparser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="checkpoint directory"
    )
    argparser.add_argument(
        "--std",
        type=float,
        default=0.2,
        help="Projection layer random initialization std",
    )
    argparser.add_argument("--wandb_name", type=str, default=None, help="wandb name")
    argparser.add_argument(
        "--wandb_project", type=str, default=None, help="wandb project"
    )
    return argparser.parse_args()


def get_epoch(filename):
    match = re.search(r"model_epoch_(\d+)\.pt", filename)
    if match:
        return int(match.group(1))
    return -1


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    lr = args.lr
    wd = args.wd
    num_epochs = args.num_epochs
    num_linear_warmup_steps = args.num_linear_warmup_steps
    training_data_dir = args.training_data_dir
    test_data_dir = args.test_data_dir
    tool_embedding_dir = args.tool_embedding_dir
    tool_name_dir = args.tool_name_dir
    train_tool_top_k_retrieval_dir = args.train_tool_top_k_retrieval_dir
    valid_tool_top_k_retrieval_dir = args.valid_tool_top_k_retrieval_dir
    use_amp = args.use_amp
    std = args.std
    num_layer_to_freeze = args.num_layer_to_freeze
    iters_to_accumulate = args.iters_to_accumulate

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T2VPretrainedReranker(
        model_name=model_name,
        std=std,
        num_layer_to_freeze=num_layer_to_freeze,
        use_cls=True,
        use_sep=True,
    ).to(device)
    print(f"Model: {model}")

    criterion = nn.BCEWithLogitsLoss()

    num_tools_to_be_presented = args.num_tools_to_be_presented
    train_dataset = T2VDatasetQueryNT(
        data_dir=training_data_dir,
        tool_name_dir=tool_name_dir,
        tool_embedding_dir=tool_embedding_dir,
        tool_top_k_retrieval_dir=train_tool_top_k_retrieval_dir,
        is_valid=False,
        num_tools_to_be_presented=num_tools_to_be_presented,
    )
    valid_dataset = T2VDatasetQueryNT(
        data_dir=test_data_dir,
        tool_name_dir=tool_name_dir,
        tool_top_k_retrieval_dir=valid_tool_top_k_retrieval_dir,
        tool_embedding_dir=tool_embedding_dir,
        is_valid=True,
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Valid dataset size: {len(valid_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size // iters_to_accumulate,
        shuffle=True,
        collate_fn=lambda x: t2v_collator_query_nt(x, tokenizer=tokenizer),
        num_workers=4,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size // iters_to_accumulate,
        shuffle=False,
        collate_fn=lambda x: t2v_collator_query_nt(x, tokenizer=tokenizer),
        num_workers=4,
    )

    checkpoint_dir = args.checkpoint_dir or os.getcwd()
    os.makedirs(checkpoint_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # print model parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # total steps
    total_steps = len(train_loader) * num_epochs
    print(f"Total steps: {total_steps}")
    current_step = 0

    if num_linear_warmup_steps < 1:
        # This indicates that the linear warmup steps are specified as a fraction of the total steps
        num_linear_warmup_steps = int(total_steps * 0.1)
        print(f"Linear warmup steps: {num_linear_warmup_steps}")

    num_linear_warmup_steps = int(num_linear_warmup_steps * lr / 0.00025)

    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Load the most recent checkpoint if it exists
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pt")]
    if checkpoint_files:
        checkpoint_files.sort(key=get_epoch, reverse=True)
        latest_checkpoint = os.path.join(checkpoint_dir, checkpoint_files[0])
    else:
        latest_checkpoint = None

    # if latest_checkpoint:
    if latest_checkpoint:
        print(f"Loading checkpoint: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        start_epoch = checkpoint["epoch"]

        if start_epoch < num_epochs:
            print(
                f"Checkpoint is from a previous training session. Starting from epoch {start_epoch}"
            )
            start_epoch = 0
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Resuming from epoch {start_epoch}")
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    if start_epoch >= num_epochs:
        print("Already trained for the specified number of epochs.")
        return

    if args.wandb_name:
        wandb.init(project="t2v_nt_ablations", config=args, name=args.wandb_name)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        for i, batch in tqdm(
            enumerate(train_loader), desc=f"Epoch {epoch}", total=len(train_loader)
        ):
            # input_ids: [batch_size, num_tokens]
            input_ids = batch["input_ids"].to(device)
            # attention_mask: [batch_size, num_tokens]
            attention_mask = batch["attention_mask"].to(device)
            # tool_embedding: [batch_size, 1, 1536]
            tool_embedding = batch["tool_embedding"].to(device)
            # label: [batch_size]
            label = batch["label"].to(device)

            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                # output: [batch_size]
                output = model(input_ids, attention_mask, tool_embedding)
                # Compute loss
                loss = criterion(output, label)
                # Accumulate
                loss = loss / iters_to_accumulate

            # Take gradient step
            scaler.scale(loss).backward()

            if ((i + 1) % iters_to_accumulate == 0) or (
                # Handle the last set of batches if they don't perfectly divide by the accumulation number
                i == len(train_loader) - 1
                and (i + 1) % iters_to_accumulate != 0
            ):
                # Update learning rate
                if current_step < num_linear_warmup_steps:
                    new_lr = lr * current_step / num_linear_warmup_steps
                    for param_group in optimizer.param_groups:
                        param_group["lr"] = new_lr
                else:
                    lr_scheduler.step()

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                if args.wandb_name and (current_step % 10 == 0):
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": optimizer.param_groups[0]["lr"],
                        }
                    )
                current_step += 1

        with torch.no_grad():
            valid_loss = 0
            valid_accuracy = 0
            num_items = 0
            total_recalls_at_k: dict[int, float] = defaultdict(float)

            for i, batch in tqdm(
                enumerate(valid_loader),
                desc="Validation Training",
                total=len(valid_loader),
            ):
                # input_ids: [batch_size, num_tokens]
                input_ids = batch["input_ids"].to(device)
                # attention_mask: [batch_size, num_tokens]
                attention_mask = batch["attention_mask"].to(device)
                # tool_embedding: [batch_size, 1, 1536]
                tool_embedding = batch["tool_embedding"].to(device)
                # label: [batch_size]
                label = batch["label"].to(device)
                # true_tools: [batch_size, num_tools]
                true_tools = batch["true_tools"]
                # labeled_tools: [batch_size, num_tools]
                labeled_tools = batch["labeled_tools"]
                # output: [batch_size]
                output = model(input_ids, attention_mask, tool_embedding)
                # import pdb;pdb.set_trace()

                # Compute loss
                loss = criterion(output, label)
                valid_loss += loss.item()

                # prediction: [batch_size]
                prediction = torch.sigmoid(output) > 0.5
                correct = (prediction == label).sum().item()
                valid_accuracy += correct / (
                    label.shape[1] * label.shape[0]
                )  # num_tools

                num_items += label.shape[0]

                if (epoch) % 1 == 0:
                    for k in [3, 5, 7, 10, 12]:
                        total_recall = top_k_recall(
                            output, labeled_tools, true_tools, top_k=k
                        )
                        total_recalls_at_k[k] += total_recall

            wandb.log(
                {
                    "valid/loss": valid_loss / len(valid_loader),
                    "valid/accuracy": valid_accuracy / len(valid_loader),
                    "epoch": epoch,
                }
            )
            for k in total_recalls_at_k.keys():
                wandb.log(
                    {
                        "epoch": epoch,
                        f"valid/recall@{k}": total_recalls_at_k[k] / num_items,
                    }
                )
                print(f"Total recall at k@{k}: ", total_recalls_at_k[k] / num_items)

            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": valid_loss / len(valid_loader),
                    "accuracy": valid_accuracy / len(valid_loader),
                    "recalls": {
                        k: np.mean(total_recalls_at_k[k])
                        for k in total_recalls_at_k.keys()
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved at {checkpoint_path}")


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)

    gc.collect()
    torch.cuda.empty_cache()

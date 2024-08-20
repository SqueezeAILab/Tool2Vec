import argparse
import os

import torch
import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer

import wandb
from toolrag.tool_reranker.utils import set_seed


def average_pool(
    last_hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, triplets):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]


def train(args):
    dataset = torch.load(args.train_data_path)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, drop_last=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)

    # Set up the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = len(dataloader) * args.epochs
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    current_step = 0
    for epoch in tqdm.trange(args.epochs):
        wandb.log({"train/epoch": epoch}, step=current_step)
        model.train()

        for batch in tqdm.tqdm(dataloader):
            anchors, positives, negatives = batch
            anchors = [f"query: {anchor}" for anchor in anchors]
            positives = [f"query: {positive}" for positive in positives]
            negatives = [f"query: {negative}" for negative in negatives]

            batch_dict = tokenizer(
                anchors + positives + negatives,
                max_length=512,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(device)
            outputs = model(**batch_dict)
            anchor_embeddings = average_pool(
                outputs.last_hidden_state[: args.batch_size],
                batch_dict["attention_mask"][: args.batch_size],
            )
            positive_embeddings = average_pool(
                outputs.last_hidden_state[args.batch_size : 2 * args.batch_size],
                batch_dict["attention_mask"][args.batch_size : 2 * args.batch_size],
            )
            negative_embeddings = average_pool(
                outputs.last_hidden_state[2 * args.batch_size :],
                batch_dict["attention_mask"][2 * args.batch_size :],
            )

            loss = torch.nn.functional.triplet_margin_loss(
                anchor_embeddings,
                positive_embeddings,
                negative_embeddings,
                margin=args.margin,
            )

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
            if args.wandb_name and current_step % 10 == 0:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/lr": optimizer.param_groups[0]["lr"],
                    },
                    step=current_step,
                )
            current_step += 1

            if current_step % 2000 == 0:
                checkpoint_path = os.path.join(
                    args.checkpoint_dir, f"model_epoch_{current_step}.pt"
                )
                torch.save(
                    {
                        "step": current_step,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_path,
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_data_path", type=str, help="Path to the training data", required=True
    )
    parser.add_argument(
        "--model_name", type=str, help="Model name to use for training", required=True
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory to save model checkpoints",
    )
    parser.add_argument("--wandb_name", type=str, default=None, help="Name for wandb")
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs for training"
    )
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=0.01, help="weight decay")
    parser.add_argument(
        "--margin", type=float, default=1, help="margin for triplet loss"
    )
    parser.add_argument(
        "--num_linear_warmup_steps",
        type=int,
        default=100,
        help="number of linear warmup steps",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    if args.wandb_name:
        wandb.init(project="t2v", group="Q2Q_G3", config=args, name=args.wandb_name)
    train(args)

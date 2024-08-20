import gc
import os
import argparse
from collections import defaultdict

import numpy as np
import re
import torch
import torch.nn as nn
from toolrag.tool_reranker.evaluations import (
    top_k_recall,
)
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
    argparser.add_argument(
        "--checkpoint_dir", type=str, required=True, help="checkpoint directory"
    )
    argparser.add_argument("--seed", type=int, default=0, help="random seed")
    argparser.add_argument("--checkpoint_epoch", type=int, default=None, help="epoch")
    argparser.add_argument("--batch_size", type=int, default=256, help="batch size")

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
        "--valid_tool_top_k_retrieval_dir",
        type=str,
        default="",
        help="tool top k retrieval directory",
    )
    argparser.add_argument(
        "--num_tools_to_be_presented",
        type=int,
        default=64,
        help="number of tools to bepresented",
    )
    argparser.add_argument(
        "--eval_metrics_save_file",
        type=str,
        default="eval_metrics",
        help="eval metrics save file",
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
    test_data_dir = args.test_data_dir
    tool_embedding_dir = args.tool_embedding_dir
    tool_name_dir = args.tool_name_dir
    valid_tool_top_k_retrieval_dir = args.valid_tool_top_k_retrieval_dir

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = T2VPretrainedReranker(
        model_name=model_name,
        std=0.2,
        num_layer_to_freeze=0,
        use_cls=True,
        use_sep=True,
    ).to(device)
    print(f"Model: {model}")

    valid_dataset = T2VDatasetQueryNT(
        data_dir=test_data_dir,
        tool_name_dir=tool_name_dir,
        tool_top_k_retrieval_dir=valid_tool_top_k_retrieval_dir,
        tool_embedding_dir=tool_embedding_dir,
        is_valid=True,
    )

    print(f"Valid dataset size: {len(valid_dataset)}")

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: t2v_collator_query_nt(x, tokenizer=tokenizer),
        num_workers=4,
    )

    checkpoint_dir = args.checkpoint_dir or os.getcwd()
    os.makedirs(checkpoint_dir, exist_ok=True)

    # print model parameter count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params}")

    # Load the most recent checkpoint if it exists
    checkpoint_epoch = args.checkpoint_epoch
    checkpoint_file_path = os.path.join(
        checkpoint_dir, f"model_epoch_{checkpoint_epoch}.pt"
    )
    print(f"Loading checkpoint: {checkpoint_file_path}")
    checkpoint = torch.load(checkpoint_file_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
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

            # prediction: [batch_size]
            prediction = torch.sigmoid(output) > 0.5
            correct = (prediction == label).sum().item()
            valid_accuracy += correct / (label.shape[1] * label.shape[0])  # num_tools

            num_items += label.shape[0]

            for k in [3, 5, 7, 10, 12]:
                total_recall = top_k_recall(output, labeled_tools, true_tools, top_k=k)
                total_recalls_at_k[k] += total_recall

        print("Checkpoint: ", checkpoint_file_path)
        for k in total_recalls_at_k.keys():
            print(f"Total recall at k@{k}: ", total_recalls_at_k[k] / num_items)


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    train(args)

    gc.collect()
    torch.cuda.empty_cache()

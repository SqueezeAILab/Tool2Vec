import argparse
import json

import torch
from torch.utils.data import Dataset


class MLCDataset(Dataset):
    def __init__(self, train_data: list[dict], tool_to_id: dict[str, int]) -> None:
        """
        Args:
            train_data (list): List of training data.
            tool_to_id (dict): Dictionary mapping tool names to their corresponding IDs.
        """
        self.train_data = train_data
        self.tool_to_id = tool_to_id

    def __len__(self) -> int:
        return len(self.train_data)

    def __getitem__(self, idx: int) -> tuple[str, torch.LongTensor]:
        data = self.train_data[idx]
        if data["refined_instruction"]:
            instruction: str = data["refined_instruction"]
        else:
            instruction: str = data["instruction"]

        tool_ids = [self.tool_to_id[tool] for tool in data["functions"]]
        labels = [0] * len(self.tool_to_id)
        for tool_id in tool_ids:
            labels[tool_id] = 1

        return instruction, torch.LongTensor(labels)


def format_data(args) -> None:
    all_tools_path = args.all_tools_path
    with open(all_tools_path, "r") as f:
        all_tools = json.load(f)

    unique_tools = list(set(all_tools.values()))
    unique_tools.sort()

    tool_to_id = {tool: i for i, tool in enumerate(unique_tools)}

    train_data_path = args.train_data_path
    with open(train_data_path, "r") as f:
        train_data = json.load(f)

    # Create an instance of the dataset
    dataset = MLCDataset(train_data, tool_to_id)

    # save dataset to a file
    torch.save(dataset, args.output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--all_tools_path",
        type=str,
        help="Path to the file containing all tools data",
        required=True,
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        help="Path to the file containing training data",
        required=True,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="mlc_training.pt",
        help="Path to save the formatted data",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    format_data(args)

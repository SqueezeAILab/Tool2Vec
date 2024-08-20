import argparse
import json

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def test(args) -> None:
    # Load the tokenizer and model
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=args.num_labels
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

    all_tools_path = args.all_tools_path
    with open(all_tools_path, "r") as f:
        all_tools = json.load(f)

    unique_tools = list(set(all_tools.values()))
    unique_tools.sort()

    test_data_path = args.test_data_path
    with open(test_data_path, "r") as f:
        test_data = json.load(f)

    # Function to perform inference
    def predict(texts):
        # Tokenize the input texts
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Perform inference
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.sigmoid(logits).cpu().numpy()

        return predictions

    all_predicted_tools = []
    for data in test_data:
        if data["refined_instruction"]:
            instruction = data["refined_instruction"]
        else:
            instruction = data["instruction"]

        # Get top k tools by probability
        probs = predict([instruction])[0]
        sorted_tools = sorted(
            zip(unique_tools, probs), key=lambda x: x[1], reverse=True
        )
        predicted_tools = [tool for tool, _ in sorted_tools]
        all_predicted_tools.append(predicted_tools)

    for k in [3, 5, 7, 10, 12]:
        recalls = []
        num_tools = []

        for idx, data in enumerate(test_data):
            gt_tools = data["functions"]
            predicted_tools = all_predicted_tools[idx][:k]

            # calculate recall
            recall = len(set(predicted_tools).intersection(gt_tools)) / len(gt_tools)
            recalls.append(recall)
            num_tools.append(len(predicted_tools))

        print(f"Recall@{k}: {np.mean(recalls)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data_path", type=str, help="Path to the test data", required=True
    )
    parser.add_argument(
        "--all_tools_path",
        type=str,
        help="Path to the file containing all tools data",
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
    parser.add_argument("--num_labels", type=int, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    test(args)

"""
Instructions to run the script:
1. Set the following environment variables:
    - AZURE_ENDPOINT: Azure endpoint for the OpenAI service
    - AZURE_OPENAI_API_KEY: API key
    - AZURE_OPENAI_API_VERSION: API version

2. Run the script with the following arguments:
    - data_path: Path to the JSON file containing the examples to generate embeddings for
    - output_path: Path to the directory where the embeddings will be saved
    - output_file_name: Name of the output file for the embeddings
    - max_workers: Maximum number of concurrent tasks to run
    - freq: Frequency of writing generated embeddings to disk
    - model: Model to use for embedding generation (azure or openai)
    - num_data_points: Number of data points to process

3. Example:
    - cd ToolRAG (root directory of the project)
    - python tool2vec/embedding_generator.py --output_file_name numpybank_t2v.json
"""

import argparse
import concurrent.futures
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Callable, Collection, Optional, Type, TypeVar

from pkg_resources import require

from openai import AzureOpenAI
from tqdm import tqdm

from toolrag.models.e5 import E5Model
from toolrag.models.mxbai import MxbaiModel

Q = TypeVar("Q", bound=Callable[..., Any])

client = AzureOpenAI(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
)


# define a retry decorator
def retry_with_exponential_backoff(
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 10,
    no_retry_on: Optional[Collection[Type[Exception]]] = None,
) -> Callable[[Q], Q]:
    """Retry a function with exponential backoff."""

    def decorator(func: Q) -> Q:
        def wrapper(*args, **kwargs):
            # Initialize variables
            num_retries = 0
            delay = initial_delay

            error = None

            # Loop until a successful response or max_retries is hit or an exception is raised
            while num_retries <= max_retries:
                try:
                    return func(*args, **kwargs)

                # Raise exceptions for any errors specified
                except Exception as e:
                    if no_retry_on is not None and type(e) in no_retry_on:
                        raise e

                    # Sleep for the delay
                    time.sleep(delay)

                    # Increment the delay
                    delay *= exponential_base * (1 + jitter * random.random())

                    # Set the error to the last exception
                    error = e

                    # Increment retries
                    num_retries += 1

                    # logging.info(f"Retrying {func.__name__} after error: {e} (retry {num_retries} of {max_retries})")
                    print(
                        f"Retrying {func.__name__} after error: {e} (retry {num_retries} of {max_retries})"
                    )

            if error is not None:
                raise error

        return wrapper

    return decorator


@retry_with_exponential_backoff(
    max_retries=20,
)
def generate_embeddings(text, model="azure"):
    if model == "openai":
        raise NotImplementedError("OpenAI model not supported for embedding generation")
    elif model == "azure":
        return (
            client.embeddings.create(input=[text], model="text-embedding-3-small")
            .data[0]
            .embedding
        )
    elif model == "e5-small" or model == "e5-base" or model == "e5-large":
        return e5_model.embed_queries([text]).squeeze(0).detach().cpu().tolist()
    elif model == "mxbai-large":
        return mxbai_model.embed_queries([text]).squeeze(0).detach().cpu().tolist()
    else:
        raise ValueError(f"Model {model} not supported")


def process_item(
    args, data_dict: dict[str, str], debug: bool = False
) -> dict[str, Any]:
    functions = data_dict["functions"]
    refined_instruction = data_dict.get("refined_instruction", None)
    unrefined_instruction = data_dict.get("instruction", None)

    if refined_instruction is None:
        instruction = unrefined_instruction

        if unrefined_instruction is None:
            raise ValueError(
                "Either refined_instruction or instruction must be provided"
            )
    else:
        instruction = refined_instruction

    function_embedding = generate_embeddings(instruction, model=args.model)

    if debug:
        print(functions, instruction, len(function_embedding))

    return {
        "functions": functions,
        "instruction": instruction,
        "function_embedding": function_embedding,
    }


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--data_path",
        type=str,
        help="Json file of examples to generate embeddings for",
        required=True,
    )
    argparser.add_argument(
        "--output_path",
        type=str,
        help="Output path for embeddings",
        required=True,
    )
    argparser.add_argument(
        "--output_file_name",
        type=str,
        required=True,
        help="Output file name for embeddings",
    )
    argparser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum number of concurrent tasks to run",
    )
    argparser.add_argument(
        "--freq",
        type=int,
        default=20000,
        help="Frequency of writing generated embeddings to disk",
    )
    argparser.add_argument(
        "--model",
        type=str,
        default="azure",
        help="Model to use for embedding generation",
        choices=["azure", "openai", "e5-small", "e5-base", "e5-large", "mxbai-large"],
    )
    argparser.add_argument(
        "--use_checkpoint",
        action="store_true",
        help="Use checkpoint for embedding generation model (e.g. e5-base)",
    )
    argparser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to checkpoint for embedding generation model (e.g. e5-base)",
    )
    argparser.add_argument(
        "--num_data_points",
        type=int,
        default=None,
        help="Number of data points to process",
    )
    argparser.add_argument("--debug", action="store_true", help="Debug mode")

    args = argparser.parse_args()

    if args.debug:
        args.max_workers = 1
        args.num_data_points = 10
        args.freq = 5
    return args


if __name__ == "__main__":
    args = parse_args()
    data_path = Path(args.data_path)
    output_file_name = args.output_file_name
    output_path = Path(args.output_path)
    output_data_path = output_path / output_file_name

    assert output_file_name.endswith(".json") or output_file_name.endswith(
        ".jsonl"
    ), "Output file format must be json"

    if not data_path.exists():
        raise ValueError(f"Data file {data_path} does not exist")

    print("Loading data from", data_path)
    with open(data_path, "r") as f:
        data = json.load(f)

    example_base_embedding_list = []

    if args.num_data_points:
        data = data[: args.num_data_points]

    if args.model == "e5-small" or args.model == "e5-base" or args.model == "e5-large":
        print(f"Loading {args.model} model")

        if args.model == "e5-small":
            e5_model = E5Model("intfloat/e5-small-v2")
        elif args.model == "e5-large":
            e5_model = E5Model("intfloat/e5-large-v2")
        else:
            e5_model = E5Model("intfloat/e5-base-v2")

        if args.use_checkpoint:
            assert args.checkpoint_path is not None, "Checkpoint path must be provided"
            print(f"Loading checkpoint from {args.checkpoint_path}")
            e5_model.load_checkpoint(args.checkpoint_path)
    elif args.model == "mxbai-large":
        mxbai_model = MxbaiModel("mixedbread-ai/mxbai-embed-large-v1")
    else:
        print(f"Using model: {args.model}")
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_workers
    ) as executor:
        futures = [executor.submit(process_item, args, d, args.debug) for d in data]

        idx = 0
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            result = future.result()
            if result is not None:
                example_base_embedding_list.append(result)

            # Save periodically
            idx += 1

            if idx % args.freq == 0:
                print("Saving data to", output_data_path)
                with open(output_data_path, "w") as f:
                    json.dump(example_base_embedding_list, f, indent=4)

    # Final save
    print("Saving data to", output_data_path)
    with open(output_data_path, "w") as f:
        json.dump(example_base_embedding_list, f, indent=4)

    os.chmod(output_data_path, 0o777)

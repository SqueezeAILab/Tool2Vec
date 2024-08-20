import argparse
import asyncio
import json
import time
import traceback

from toolrag.data_generation.models import BaseInstruction
from toolrag.data_generation.bank_numpy.bank_numpy_instruction_generator import (
    BankNumpyInstructionGenerator,
)
from langchain_openai import ChatOpenAI

from toolrag.data_generation.bank_numpy.bank_numpy_instruction_polisher import (
    BankNumpyInstructionPolisher,
)
from toolrag.data_generation.bank_pandas.bank_pandas_instruction_generator import (
    BankPandasInstructionGenerator,
)
from toolrag.data_generation.bank_pandas.bank_pandas_instruction_polisher import (
    BankPandasInstructionPolisher,
)
from toolrag.data_generation.bank_aws.bank_aws_instruction_generator import (
    BankAWSInstructionGenerator,
)
from toolrag.data_generation.bank_aws.bank_aws_instruction_polisher import (
    BankAWSInstructionPolisher,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["numpy", "pandas", "aws"],
        help="Name of the dataset to generate",
    )
    parser.add_argument(
        "--iterations", type=int, default=5, help="Number of iterations to run"
    )
    parser.add_argument(
        "--max_instructions", type=int, default=None, help="Max number of instructions"
    )

    # Model parameters
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for inference server (if running locally)",
    )

    # Instruction generation parameters
    parser.add_argument(
        "--tools_path",
        type=str,
        required=True,
        help="Path to the tools json file",
    )
    parser.add_argument(
        "--max_concurrent_tasks",
        type=int,
        default=1,
        help="Number of concurrent tasks to run. For faster batched inference.",
    )
    parser.add_argument(
        "--n_functions_to_sample",
        type=int,
        default=10,
        help="Number of functions to sample per iteration",
    )
    parser.add_argument(
        "--n_subset_to_choose",
        type=int,
        nargs="+",
        default=[3],
        help="Number of functions to choose from the sampled functions",
    )
    parser.add_argument(
        "--n_in_context_examples",
        type=int,
        default=5,
        help="Number of in context examples to use for generation",
    )
    parser.add_argument(
        "--use_polisher",
        action="store_true",
        help="Use the instruction polisher to polish the generated instructions",
        default=False,
    )

    # Seed data for initializing the in context example generators
    parser.add_argument(
        "--seed_data",
        type=str,
        default=None,
        help="Path to seed data (if any) to use for in context examples",
    )

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="Path to save the generated instructions",
    )
    return parser.parse_args()


async def main():
    args = parse_args()

    if args.port is not None:
        inference_server_url = f"http://localhost:{args.port}/v1"
        instruction_generator_model = ChatOpenAI(
            model=args.model,
            openai_api_key="EMPTY",  # type: ignore
            openai_api_base=inference_server_url,  # type: ignore
            max_tokens=4096,
            temperature=0.9,
        )
    else:
        instruction_generator_model = ChatOpenAI(
            name=args.model,
            verbose=False,
            temperature=0.9,
        )

    if args.dataset == "numpy":
        print("Using Numpy dataset...")
        instruction_polisher = BankNumpyInstructionPolisher(instruction_generator_model)
        instruction_generator = BankNumpyInstructionGenerator(
            instruction_generator_model,
            instruction_polisher,
            args.tools_path,
            args.seed_data,
        )
    elif args.dataset == "pandas":
        print("Using Pandas dataset...")
        instruction_polisher = BankPandasInstructionPolisher(
            instruction_generator_model
        )
        instruction_generator = BankPandasInstructionGenerator(
            instruction_generator_model,
            instruction_polisher,
            args.tools_path,
            args.seed_data,
        )
    elif args.dataset == "aws":
        print("Using AWS dataset...")
        instruction_polisher = BankAWSInstructionPolisher(instruction_generator_model)
        instruction_generator = BankAWSInstructionGenerator(
            instruction_generator_model,
            instruction_polisher,
            args.tools_path,
            args.seed_data,
        )

    all_numpy_instructions = []
    for i in range(args.iterations):
        if len(all_numpy_instructions) >= args.max_instructions:
            print(
                f"Stopping early as max instructions {args.max_instructions} reached..."
            )
            break

        try:
            print("=" * 30)
            print(f"Iteration {i + 1}/{args.iterations}")
            start = time.time()
            instructions: list[BaseInstruction] = []
            for subset in args.n_subset_to_choose:
                print(f"Generating instructions for subset {subset}...")
                instructions.extend(
                    await instruction_generator.generate_instructions(
                        max_concurrent_tasks=args.max_concurrent_tasks,
                        n_functions_to_sample=args.n_functions_to_sample,
                        n_subset_to_choose=subset,
                        n_in_context_examples=args.n_in_context_examples,
                        use_polisher=args.use_polisher,
                    )
                )
            end = time.time()
            print(
                f"Finished generating {len(instructions)} tools and instructions in {end - start:.2f} seconds."
            )
        except Exception as e:
            traceback.print_exc()
            print(f"Error generating numpy instructions: {e}")
            instructions = []

        all_numpy_instructions.extend(
            {"iteration": i, **instruction.to_json()} for instruction in instructions
        )

        with open(args.save_path, "w") as f:
            json.dump(all_numpy_instructions, f, indent=4)

    # Ensure all instructions were saved
    with open(args.save_path, "w") as f:
        json.dump(all_numpy_instructions, f, indent=4)


if __name__ == "__main__":
    asyncio.run(main())

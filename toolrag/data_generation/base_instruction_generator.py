import asyncio
import json
from abc import ABC

from toolrag.data_generation.in_context_example_repo import InContextExampleRepo
from toolrag.data_generation.instruction_polisher import InstructionPolisher
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from toolrag.data_generation.models import BaseInstruction
from toolrag.data_generation.tool_sampler import ToolSampler


class BaseInstructionGenerator(ABC):
    GENERIC_INSTRUCTION_GENERATOR_SYSTEM_PROMPT = (
        "You are an expert in utilizing a library of functions and generating diverse scenarios where a set of selected functions are applied to solve real-world problems. "
        "You will be provided with a set of functions and their descriptions, and will be tasked with selecting a subset of these functions to craft detailed scenerios. "
        "You will generate clear and detailed user instructions, list the names of the relevant functions, and explain how these functions can be applied to complete the task. "
        "These tasks should demonstrate a wide range of functionalities and real-life applications to ensure variety and utility.\n"
        "Guidelines:\n"
        "- The instructions must be clear and comprehensive, allowing users to understand how to apply the functions without ambiguity. "
        "However, the instructions shouldn't be robotic and shouldn't sound like 'step-by-step' instructions. For example "
        """instead of writing "Calculate the non-negative square root of an array element-wise, then round the resulting array to the nearest even value, and return the indices that would sort the array along a specified axis." which breaks down each step mechanically, you MUST instead write a more natural and fluid instruction like "Sort the array along a specified axis after calculating the non-negative square root of each element and rounding the result to the nearest even value." """
        "- You MUST select and sequence the functions in a way that demonstrates their interdependency. Ideally, a function's output should be the input to another function (or multiple functions), creating a chain of operations that solve the task at hand. In other words, the functions you select must not be selected randomly but instead be used to solve coherent multi-step problems. "
        "- The explanations should logically connect the functions to the tasks, demonstrating the workflow clearly. "
        "- Your response should be returned as a single JSON object, representing a unique user instruction. Diversity in function use and application context is crucial; avoid repetition of similar tasks or functional applications to ensure a broad coverage of the capabilities of the functions. "
        "Here is an example output of a list of JSON objects representing very distinct and detailed tasks:\n"
        "[{examples_str}]\n"
        "You MUST only return a single JSON object - do not add any extra text before and after the json object. "
        "The instructions that you generate MUST be very diverse and distinct from each other and MUST be as different as possible from the examples above. "
        "{library_specific_instructions}"
    )

    GENERIC_INSTRUCTION_GENERATOR_HUMAN_PROMPT = (
        "Here are {n_functions} functions sampled from a library of functions:\n{functions_str}\n"
        "You are tasked with selecting {n_subset} functions from the list above and generating an instruction (in form of a JSON object) where these functions are used together to solve a problem."
    )

    _model: ChatOpenAI
    _in_context_example_repo: InContextExampleRepo[BaseInstruction]
    _tool_sampler: ToolSampler
    _instruction_polisher: InstructionPolisher

    def __init__(
        self,
        model: ChatOpenAI,
        instruction_polisher: InstructionPolisher,
        tools_path: str,
        in_context_examples_path: str | None,
    ) -> None:
        self._model = model
        self._instruction_polisher = instruction_polisher
        self._tool_sampler = ToolSampler(tools_path)

        in_context_examples = self.initial_seed_data
        if in_context_examples_path is not None:
            with open(in_context_examples_path, "r") as f:
                data = json.load(f)
                in_context_examples.extend([BaseInstruction.from_json(d) for d in data])

        self._in_context_example_repo = InContextExampleRepo[BaseInstruction](
            in_context_examples
        )

    @property
    def initial_seed_data(self) -> list[BaseInstruction]:
        """Library specific initial seed data. Can be just an empty list."""
        return []

    @property
    def library_specific_instructions(self) -> str:
        """Library specific instructions for generating instructions. Can be an empty string."""
        return ""

    async def generate_instructions(
        self,
        max_concurrent_tasks: int,
        n_functions_to_sample: int,
        n_subset_to_choose: int,
        n_in_context_examples: int,
        use_polisher: bool,
    ) -> list[BaseInstruction]:
        tasks = []
        for _ in range(max_concurrent_tasks):
            in_context_examples = [
                example.to_example()
                for example in self._in_context_example_repo.get_examples(
                    n_in_context_examples
                )
            ]
            examples_str = ",\n".join(in_context_examples)

            tools = self._tool_sampler.sample_tools(n_functions_to_sample)
            functions_str = "\n\n".join(
                [
                    f"{i+1}. {tool.func_name}: {tool.description}"
                    for i, tool in enumerate(tools)
                ]
            )

            messages = [
                SystemMessage(
                    content=BaseInstructionGenerator.GENERIC_INSTRUCTION_GENERATOR_SYSTEM_PROMPT.format(
                        examples_str=examples_str,
                        library_specific_instructions=self.library_specific_instructions,
                    )
                ),
                HumanMessage(
                    content=BaseInstructionGenerator.GENERIC_INSTRUCTION_GENERATOR_HUMAN_PROMPT.format(
                        n_functions=n_functions_to_sample,
                        functions_str=functions_str,
                        n_subset=n_subset_to_choose,
                    )
                ),
            ]

            tasks.append(self._model.ainvoke(messages))

        gpt_outputs = await asyncio.gather(*tasks)

        instructions = []
        for output in gpt_outputs:
            instructions.extend(
                self._parse_gpt_output_into_instructions(output.content)
            )

        # Filter out hallucinated instructions
        instructions = self._filter_hallucination(instructions)

        # Polish the instructions
        if use_polisher:
            instructions = await self._polish_instructions(instructions)

        # Add them to the in-context example repo
        self._in_context_example_repo.add_examples(instructions)

        return instructions

    def _parse_gpt_output_into_instructions(
        self, gpt_output: str
    ) -> list[BaseInstruction]:
        gpt_output = gpt_output.strip()
        gpt_output = gpt_output[gpt_output.find("{") : gpt_output.rfind("}") + 1]
        try:
            json_object = json.loads(gpt_output)
        except json.JSONDecodeError:
            return []

        json_object["refined_instruction"] = None
        return [BaseInstruction.from_json(json_object)]

    def _filter_hallucination(
        self, instructions: list[BaseInstruction]
    ) -> list[BaseInstruction]:
        """If a tool name was hallucinated, filter it out."""
        available_tools = [tool.func_name for tool in self._tool_sampler.tools]
        return [
            instruction
            for instruction in instructions
            if all(function in available_tools for function in instruction.functions)
        ]

    async def _polish_instructions(
        self, instructions: list[BaseInstruction]
    ) -> list[BaseInstruction]:
        polished_instruction_tasks = []
        for instruction in instructions:
            polished_instruction_tasks.append(
                self._instruction_polisher.polish_instruction(instruction)
            )

        polished_instructions = await asyncio.gather(*polished_instruction_tasks)

        return polished_instructions

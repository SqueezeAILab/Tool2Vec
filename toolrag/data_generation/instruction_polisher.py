import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from toolrag.data_generation.models import (
    BaseInstruction,
    PolishedInstruction,
    PolishedInstructionICE,
)
from abc import ABC, abstractmethod


class InstructionPolisher(ABC):
    INSTRUCTION_POLISHER_SYSTEM_PROMPT = (
        "You are an expert at refining user instructions to make them more coherent and less robotic. "
        "You will be given a user instruction and will be tasked to refine the user instruction if it:\n"
        "- Sounds too robotic or step-by-step like saying 'Do this, do that, and then do this'. In other words, the instructions shouldn't break down each step mechanically but be more fluid."
        "For example "
        """instead of writing "Analyze the lyrics of the song 'XYZ', generate a playlist based on the emotions and themes found, and create a Spotify playlist with the recommended songs." """
        """you would write "Create a Spotify playlist based on the emotions and themes found in the lyrics of the song 'XYZ'."\n"""
        "- Has conditional statements like 'if this, then do that' or 'when this happens, do that'. It should be more direct and non-conditional.\n"
        "If none of the above applies to an instruction, you should mark it as good, and provide a reasoning for why it is good. "
        "Here example outputs of a JSON object representing a refined user instruction:\n"
        "{ice_str}\n"
        "You MUST only return the one json object - do not add any extra text before and after json object. "
    )

    INSTRUCTION_POLISHER_HUMAN_PROMPT = "Input: '{instruction}'\nOutput:\n"

    _model: ChatOpenAI

    def __init__(self, model: ChatOpenAI):
        self._model = model

    @property
    @abstractmethod
    def in_context_examples(self) -> list[PolishedInstructionICE]:
        pass

    async def polish_instruction(self, instruction: BaseInstruction) -> BaseInstruction:
        messages = [
            SystemMessage(
                content=InstructionPolisher.INSTRUCTION_POLISHER_SYSTEM_PROMPT.format(
                    ice_str="###\n".join(
                        [ice.to_str() for ice in self.in_context_examples]
                    )
                )
            ),
            HumanMessage(
                content=InstructionPolisher.INSTRUCTION_POLISHER_HUMAN_PROMPT.format(
                    instruction=instruction.instruction
                )
            ),
        ]

        response = await self._model.ainvoke(messages)
        polished_instruction = self._parse_gpt_response(response.content)  # type: ignore

        if polished_instruction is None:
            return instruction

        # Update the instruction with the refined instruction
        instruction.polished_instruction = polished_instruction.instruction
        return instruction

    def _parse_gpt_response(self, response: str) -> PolishedInstruction | None:
        try:
            json_response = json.loads(response)
        except json.JSONDecodeError:
            return None

        return PolishedInstruction(
            reasoning=json_response["reasoning"],
            instruction=json_response["instruction"],
        )

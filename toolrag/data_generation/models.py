import json
from dataclasses import dataclass
from typing import Any


@dataclass
class Tool:
    func_name: str
    description: str


@dataclass
class BaseInstruction:
    instruction: str
    polished_instruction: str | None
    functions: list[str]
    explanation: str

    @classmethod
    def from_json(cls, json_obj: dict[str, Any]) -> "BaseInstruction":
        return cls(
            instruction=json_obj["instruction"],
            polished_instruction=json_obj["refined_instruction"],
            functions=json_obj["functions"],
            explanation=json_obj["explanation"],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "instruction": self.instruction,
            "refined_instruction": self.polished_instruction,
            "functions": self.functions,
            "explanation": self.explanation,
        }

    def to_example(self) -> str:
        return (
            "{\n"
            f'    "instruction": "{self.polished_instruction or self.instruction}",\n'
            f'    "functions": {json.dumps(self.functions)},\n'
            f'    "explanation": "{self.explanation}"\n'
            "}"
        )


@dataclass
class PolishedInstruction:
    reasoning: str
    instruction: str


@dataclass
class PolishedInstructionICE:
    input: str
    output: PolishedInstruction

    def to_str(self) -> str:
        return (
            f"Input: '{self.input}'\n"
            "Output:\n"
            "{\n"
            f'    "reasoning": "{self.output.reasoning}",\n'
            f'    "instruction": "{self.output.instruction}"\n'
            "}\n"
        )

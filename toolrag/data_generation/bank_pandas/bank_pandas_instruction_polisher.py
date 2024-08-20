from toolrag.data_generation.models import (
    PolishedInstruction,
    PolishedInstructionICE,
)
from toolrag.data_generation.instruction_polisher import InstructionPolisher


class BankPandasInstructionPolisher(InstructionPolisher):
    @property
    def in_context_examples(self):
        return [
            PolishedInstructionICE(
                input="Capitalize all textual elements in a series, encode them uniquely, and then convert the series to a tabular format.",
                output=PolishedInstruction(
                    reasoning="The instruction is too robotic and is step-by-step. It breaks down each task in a mechanical way. It should be more fluid and natural",
                    instruction="Convert a series of all textual elements to a tabular format after capitalizing and encoding them uniquely.",
                ),
            ),
            PolishedInstructionICE(
                input="For a given dataframe, filter columns based on specific criteria, apply a transformation function to these columns, and then calculate the sum of each row.",
                output=PolishedInstruction(
                    reasoning="The instruction is too robotic and is step-by-step. It breaks down each task in a mechanical way. It should be more fluid and natural",
                    instruction="Transform and sum the values of specific columns in a dataframe based on filtering criteria.",
                ),
            ),
        ]

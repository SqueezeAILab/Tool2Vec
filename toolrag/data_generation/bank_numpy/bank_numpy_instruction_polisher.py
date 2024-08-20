from toolrag.data_generation.models import (
    PolishedInstruction,
    PolishedInstructionICE,
)
from toolrag.data_generation.instruction_polisher import InstructionPolisher


class BankNumpyInstructionPolisher(InstructionPolisher):
    @property
    def in_context_examples(self):
        return [
            PolishedInstructionICE(
                input="Given a dataset containing the monthly revenue for a company, calculate the cumulative revenue for each month, and then append the annual total revenue at the end of the array.",
                output=PolishedInstruction(
                    reasoning="The instruction is too robotic and is step-by-step. It breaks down each task in a mechanical way. It should be more fluid and natural",
                    instruction="Calculate the cumulative revenue for each month in the dataset and append the annual total revenue at the end of the array.",
                ),
            ),
            PolishedInstructionICE(
                input="Calculate the minimum value excluding NaNs from two arrays, deduct the resulting minimum from the exponentiation of the two arrays, and round the cube-root of the final array to the nearest integer.",
                output=PolishedInstruction(
                    reasoning="The instruction is too robotic and is step-by-step. It breaks down each task in a mechanical way. It should be more fluid and natural",
                    instruction="Calculate the cube-root of the result of the exponentiation of two arrays after deducting the minimum value, excluding NaNs, from the two arrays and round the result to the nearest integer.",
                ),
            ),
        ]

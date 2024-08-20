from toolrag.data_generation.models import (
    PolishedInstruction,
    PolishedInstructionICE,
)
from toolrag.data_generation.instruction_polisher import InstructionPolisher


class BankAWSInstructionPolisher(InstructionPolisher):
    @property
    def in_context_examples(self):
        return [
            PolishedInstructionICE(
                input="Create a new IAM user, grant the user S3 access, and upload a file to an S3 bucket.",
                output=PolishedInstruction(
                    reasoning="The instruction is too robotic and step-by-step. It breaks down each task in a mechanical way. It should be more fluid and natural.",
                    instruction="Create a new IAM user with S3 access and upload a file to an S3 bucket.",
                ),
            ),
            PolishedInstructionICE(
                input="Migrate a DB instance to a new region, update the instance's attributes, and create a platform endpoint for push notifications.",
                output=PolishedInstruction(
                    reasoning="The instruction is too robotic and step-by-step. It breaks down each task in a mechanical way. It should be more fluid and natural.",
                    instruction="Migrate the DB instance to a new region with updated attributes and create a platform endpoint for push notifications.",
                ),
            ),
        ]

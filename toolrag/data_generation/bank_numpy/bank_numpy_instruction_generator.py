from toolrag.data_generation.base_instruction_generator import BaseInstructionGenerator
from toolrag.data_generation.models import BaseInstruction
from typing_extensions import override


class BankNumpyInstructionGenerator(BaseInstructionGenerator):
    @property
    @override
    def initial_seed_data(self) -> list[BaseInstruction]:
        return [
            # 2 function example
            BaseInstruction(
                instruction="Simulate the roll of a die 1000 times and count the occurrence of each number.",
                polished_instruction=None,
                functions=["random_roll", "count_occurrences"],
                explanation="Use random_roll to simulate the dice rolls, generating numbers from 1 to 6. Then, use count_occurrences to count the occurrence of each number in the results.",
            ),
            # 4 function example
            BaseInstruction(
                instruction="Create a signal interference pattern created by two sine waves of different frequencies.",
                polished_instruction=None,
                functions=[
                    "linear_interval",
                    "sin_wave",
                    "pi_constant",
                    "sum_elements",
                ],
                explanation="Generate time points using linear_interval, then create two sine waves with sin_wave and pi_constant. Combine these waves using sum_elements to simulate interference.",
            ),
            # 3 function example
            BaseInstruction(
                instruction="Given a 1D array of time series data, resample the data to a certain frequency using Fourier transform.",
                polished_instruction=None,
                functions=["frequency_spectrum", "fourier_transform", "real_part"],
                explanation="Use frequency_spectrum to get the frequencies corresponding to the data. Then, apply fourier_transform to perform the Fourier transform. Resample the data by selecting the desired frequency range and applying real_part to get the real part of the transformed data.",
            ),
        ]

    @property
    @override
    def library_specific_instructions(self) -> str:
        return (
            "For example, if there is an example above about signal processing, you should not generate another example about signal processing. Similarly, if there is an example above or if you have already generated "
            "an example using the aggregate_sum, you should not generate another example using aggregate_sum. "
            "Unless the instruction explicitly states or requires to 'create' an array, assume that the array is already given so "
            "that you don't have to use functions like assemble_array to generate the array. Refrain fro using phrases like 'Given a dataset/array/list...' but instead embed the context in the instruction. "
        )

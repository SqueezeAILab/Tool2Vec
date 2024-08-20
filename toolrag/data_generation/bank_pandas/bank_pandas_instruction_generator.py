from toolrag.data_generation.base_instruction_generator import BaseInstructionGenerator
from toolrag.data_generation.models import BaseInstruction
from typing_extensions import override


class BankPandasInstructionGenerator(BaseInstructionGenerator):
    @property
    @override
    def initial_seed_data(self) -> list[BaseInstruction]:
        return [
            # 2 function example
            BaseInstruction(
                instruction="Calculate the discrete deltas in a dataset and return the smallest 5 values.",
                polished_instruction=None,
                functions=["calculate_discrete_deltas", "least_values"],
                explanation="First, use calculate_discrete_deltas to compute the differences between consecutive elements. Then, use least_values to find the 5 smallest delta values.",
            ),
            # 4 function example
            BaseInstruction(
                instruction="Convert a list of dates to timestamps and determine if any are quarter-end dates.",
                polished_instruction=None,
                functions=[
                    "convert_to_timestamp",
                    "terminates_periodic_quarter",
                    "invoke_custom_function",
                    "index_to_tabular",
                ],
                explanation="Convert the list of dates to timestamps using convert_to_timestamp. Check if any date marks the end of a quarter with terminates_periodic_quarter. Use invoke_custom_function to apply a custom filtering function, and finally, format the results using index_to_tabular.",
            ),
            # 3 function example
            BaseInstruction(
                instruction="Convert a series of all textual elements to a tabular format after capitalizing and encoding them uniquely.",
                polished_instruction=None,
                functions=[
                    "capitalize_text",
                    "enumerated_encoder",
                    "series_to_tabular",
                ],
                explanation="First, use capitalize_text to transform all text in the series to uppercase. Next, apply enumerated_encoder to assign a unique integer code to each distinct uppercase text. Finally, transform the encoded series into a tabular data structure using series_to_tabular.",
            ),
        ]

    @property
    @override
    def library_specific_instructions(self) -> str:
        return (
            "For example, if there is an example above about time series, you should not generate another example about time series. Similarly, if there is an example above or if you have already generated "
            "an example using the index_to_tabular, you should not generate another example using index_to_tabular. "
            "Unless the instruction explicitly states or requires to 'create' an array or dataset, assume that the array is already given so "
            "that you don't have to use functions to generate the array. Refrain from using phrases like 'Given a dataset/array/list...' but instead embed the context in the instruction. "
        )

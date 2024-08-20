import json
import random

from toolrag.data_generation.models import Tool


class ToolSampler:
    tools: list[Tool]

    def __init__(self, tools_file: str) -> None:
        self.tools = []
        with open(tools_file, "r") as f:
            tools = json.load(f)
            for _, tool in tools.items():
                if "new_func" not in tool or "description" not in tool:
                    raise ValueError(
                        "Each tool should have a 'func_name' and 'description' field."
                    )
                self.tools.append(Tool(tool["new_func"], tool["description"]))

    def sample_tools(self, num_samples: int = 20) -> list[Tool]:
        if num_samples > len(self.tools):
            raise ValueError(
                "Not enough tools to sample the requested number of tools."
            )
        return random.sample(self.tools, num_samples)

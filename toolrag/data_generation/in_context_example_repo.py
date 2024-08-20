import random
from typing import Generic, TypeVar

T = TypeVar("T")


class InContextExampleRepo(Generic[T]):
    _examples: list[T]

    def __init__(self, seed_data: list[T]) -> None:
        self._examples = seed_data

    def get_examples(self, n: int) -> list[T]:
        return random.sample(self._examples, min(n, len(self._examples)))

    def add_examples(self, examples: list[T]) -> None:
        self._examples.extend(examples)

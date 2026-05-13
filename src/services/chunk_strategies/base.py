from abc import ABC, abstractmethod
from typing import Any


class BaseChunkStrategy(ABC):
    @property
    @abstractmethod
    def strategy_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        raise NotImplementedError

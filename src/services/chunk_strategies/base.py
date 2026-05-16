"""所有切块策略实现都必须遵守的抽象契约。"""

from abc import ABC, abstractmethod
from typing import Any


class BaseChunkStrategy(ABC):
    """为不同切块策略提供统一接口。"""

    @property
    @abstractmethod
    def strategy_type(self) -> str:
        """返回策略类型标识，例如 RCTS、SC、JE。"""
        raise NotImplementedError

    @abstractmethod
    def split(self, text: str, **kwargs) -> list[dict[str, Any]]:
        """执行文本切分，并返回统一格式的 chunk payload 列表。"""
        raise NotImplementedError

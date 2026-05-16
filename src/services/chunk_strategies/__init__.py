"""切块策略选择工厂。"""

import logging

from services.chunk_strategies.je_strategy import JEStrategy
from services.chunk_strategies.rcts_strategy import RCTSStrategy
from services.chunk_strategies.sc_strategy import SCStrategy


logger = logging.getLogger(__name__)


def build_strategy(strategy_type, config, embedding_service):
    """根据策略类型实例化对应的切块策略。"""
    strategy_type = strategy_type.upper()
    logger.info("Building chunk strategy type=%s", strategy_type)
    if strategy_type == "RCTS":
        return RCTSStrategy()
    if strategy_type == "SC":
        return SCStrategy(embedding_service)
    if strategy_type == "JE":
        return JEStrategy(config)
    raise ValueError(f"Unsupported chunk type: {strategy_type}")

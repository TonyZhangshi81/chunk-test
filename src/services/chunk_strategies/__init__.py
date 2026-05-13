from services.chunk_strategies.je_strategy import JEStrategy
from services.chunk_strategies.rcts_strategy import RCTSStrategy
from services.chunk_strategies.sc_strategy import SCStrategy


def build_strategy(strategy_type, config, embedding_service):
    strategy_type = strategy_type.upper()
    if strategy_type == "RCTS":
        return RCTSStrategy()
    if strategy_type == "SC":
        return SCStrategy(embedding_service)
    if strategy_type == "JE":
        return JEStrategy(config)
    raise ValueError(f"Unsupported chunk type: {strategy_type}")

"""应用级日志配置辅助工具。"""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> None:
    """为 CLI 进程一次性配置根日志记录器。"""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

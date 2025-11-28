"""
Structured logging configuration using Loguru.

This module provides consistent logging setup across the application
with file rotation and structured output.
"""

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_dir: Path | None = None,
    rotation: str = "10 MB",
    retention: str = "1 week",
) -> None:
    """
    Configure structured logging for the application.

    Parameters
    ----------
    level : str, optional
        Minimum log level, defaults to "INFO"
    log_dir : Path, optional
        Directory for log files, defaults to "logs"
    rotation : str, optional
        Log rotation size, defaults to "10 MB"
    retention : str, optional
        Log retention period, defaults to "1 week"
    """
    logger.remove()

    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    logger.add(
        sys.stderr,
        format=log_format,
        level=level,
        colorize=True,
    )

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        file_format = (
            "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}"
        )

        logger.add(
            log_dir / "geo-rect.log",
            format=file_format,
            level=level,
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

        logger.add(
            log_dir / "errors.log",
            format=file_format,
            level="ERROR",
            rotation=rotation,
            retention=retention,
            compression="zip",
        )

    logger.info(f"Logging configured with level={level}")

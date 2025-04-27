"""Logging configuration for TSNE-PSO.

This module provides centralized logging configuration and utilities.
"""

import logging
import sys
from typing import Optional


def setup_logging(level: int = logging.INFO) -> None:
    """Set up logging configuration for the entire package.

    Args:
        level: The logging level to use. Defaults to INFO.
    """
    # Create root logger
    root_logger = logging.getLogger("tsne_pso")
    root_logger.setLevel(level)

    root_logger.handlers.clear()

    # Create console handler with formatter
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    """Get a logger instance with the specified name.

    Args:
        name: The name of the logger, typically __name__
        level: Optional specific level for this logger

    Returns:
        A configured logger instance
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger


# Set up default logging configuration
setup_logging()

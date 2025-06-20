"""
Logging utilities for the 3D nuclei detection project
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


def setup_logger(
    name: Optional[str] = None,
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
    include_timestamp: bool = True,
) -> logging.Logger:
    """
    Set up a logger with console and file handlers

    Args:
        name: Logger name (uses root logger if None)
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Path to log file (console only if None)
        format_string: Custom format string
        include_timestamp: Whether to include timestamp in log file name

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    if format_string is None:
        format_string = "[%(asctime)s] %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_file = Path(log_file)

        # Add timestamp to filename if requested
        if include_timestamp and not log_file.name.endswith(".log"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = log_file.parent / f"{log_file.stem}_{timestamp}.log"

        # Create directory if it doesn't exist
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Backwards compatibility
def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None, log_to_console: bool = True
) -> logging.Logger:
    """
    Legacy setup logging function for backwards compatibility
    """
    return setup_logger(
        level=log_level,
        log_file=log_file if log_file else None,
        include_timestamp=False,
    )


class Logger:
    """
    Custom logger wrapper for training
    """

    def __init__(self, name: str = "CentroidModel", log_level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))

        # Prevent duplicate logs
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)

    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)

    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)

    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)

    def critical(self, message: str):
        """Log critical message"""
        self.logger.critical(message)

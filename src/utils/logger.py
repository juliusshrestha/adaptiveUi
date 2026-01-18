"""
Logging configuration for Adaptive UI System
Provides centralized logging with file and console handlers
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "adaptive_ui",
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    console_output: bool = True
) -> logging.Logger:
    """
    Setup logger with file and console handlers

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (default: 'logs')
        log_file: Log file name (default: 'adaptive_ui_YYYYMMDD.log')
        console_output: Whether to output to console

    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # Clear any existing handlers
    logger.handlers.clear()

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_dir or log_file:
        # Setup log directory
        log_directory = Path(log_dir) if log_dir else Path("logs")
        log_directory.mkdir(parents=True, exist_ok=True)

        # Setup log file name
        if not log_file:
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = f"adaptive_ui_{timestamp}.log"

        log_path = log_directory / log_file

        # Create file handler
        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    return logger


def get_logger(name: str = "adaptive_ui") -> logging.Logger:
    """
    Get existing logger instance

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)

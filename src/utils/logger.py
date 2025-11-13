"""
Centralized logging configuration for the trading platform.

Uses loguru for better log formatting and automatic rotation.
Logs are structured for easy parsing and debugging.
"""

from loguru import logger
import sys
from pathlib import Path
from src.utils.config import settings


def setup_logger():
    """
    Configure the application logger with file and console outputs.

    Benefits:
    - Automatic log rotation to prevent disk space issues
    - Colored output for better readability
    - Structured logging with context
    - Separate log levels for file and console
    """

    # Remove default handler
    logger.remove()

    # Console handler with color
    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level=settings.log_level,
    )

    # File handler with rotation
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger.add(
        settings.log_file,
        rotation="500 MB",  # Rotate when file reaches 500MB
        retention="30 days",  # Keep logs for 30 days
        compression="zip",  # Compress old logs
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # More verbose in files
        enqueue=True,  # Thread-safe
    )

    # Separate file for errors
    logger.add(
        "logs/errors.log",
        rotation="100 MB",
        retention="90 days",
        compression="zip",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
    )

    logger.info("Logger initialized successfully")
    return logger


# Initialize logger
app_logger = setup_logger()

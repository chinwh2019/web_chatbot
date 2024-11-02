import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional

from ..settings import settings


class CustomFormatter(logging.Formatter):
    """Custom formatter with color support"""

    grey = "\x1b[38;21m"
    blue = "\x1b[38;5;39m"
    yellow = "\x1b[38;5;226m"
    red = "\x1b[38;5;196m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: grey
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.INFO: blue
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.WARNING: yellow
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.ERROR: red
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
        logging.CRITICAL: bold_red
        + "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Set up logger with both file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)

    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomFormatter())
    logger.addHandler(console_handler)

    # File Handler (if log_file is specified)
    if log_file:
        log_path = Path(settings.LOG_DIR) / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=settings.LOG_MAX_SIZE,
            backupCount=settings.LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


# Default logger instance
logger = setup_logger("web_chatbot", "app.log")

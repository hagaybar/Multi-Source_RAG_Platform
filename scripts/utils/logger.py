"""
This module provides a centralized and configurable logging system.
It features `LoggerManager` for creating and managing logger instances with
various handlers (file, console) and formats (text, colored, JSON), and
`JsonLogFormatter` for producing structured JSON logs.
"""

import os
import sys
import logging
import json
from typing import Optional

try:
    from colorlog import ColoredFormatter

    COLORLOG_AVAILABLE = True
except ImportError:
    COLORLOG_AVAILABLE = False


class LoggerManager:
    """
    A factory class for creating and managing singleton `logging.Logger` instances.

    This class uses a class method `get_logger` to provide configured logger
    instances. It ensures that for any unique logger name (optionally appended
    with a `run_id`), the same logger instance is returned, preventing duplicate
    handler attachments.

    Key features of the configured loggers:
    - **Dual Output**: Loggers are set up with handlers for both console (stdout)
      and file output.
    - **Customizable Levels & Formatting**: The logging level (e.g., "INFO", "DEBUG")
      can be specified. Console output can be colored if the `colorlog` package
      is available. File output can be formatted as plain text or structured JSON
      (using `JsonLogFormatter`).
    - **Path Handling**: Log file paths can be specified directly, derived from a
      `TaskPaths` object (if provided, allowing for task-specific log organization,
      potentially with `run_id` subdirectories), or fall back to a default
      directory. The class ensures necessary log directories are created.
    - **No Duplicate Propagation**: Logger propagation is disabled
      (`propagate = False`)
      to avoid messages being handled multiple times by ancestor loggers.

    The `_loggers` class attribute stores references to already created loggers
    to ensure their singleton nature.
    """

    _loggers = {}
    _default_log_dir = "logs"

    @classmethod
    def get_logger(
        cls,
        name: str,
        log_file: Optional[str] = None,
        level: str = "DEBUG",
        use_json: bool = False,
        use_color: bool = True,
        task_paths: Optional[object] = None,
        run_id: Optional[str] = None,
    ) -> logging.Logger:
        """
        Retrieve or create a logger configured for console and file output.

        Args:
            name (str): A unique identifier (typically module or task name).
            log_file (Optional[str]): Full path to a log file. Overrides
                task_paths if set.
            level (str): Logging level threshold ("DEBUG", "INFO", etc.).
            use_json (bool): If True, format file logs as JSON (for parsing).
            use_color (bool): If True and colorlog is available, enable colored
                console output.
            task_paths (Optional[object]): An instance of TaskPaths to resolve
                log file path.
            run_id (Optional[str]): Optional run identifier used to create
                per-run logs.

        Returns:
            logging.Logger: A fully configured logger instance.
        """

        logger_key = f"{name}-{run_id}" if run_id else name
        if logger_key in cls._loggers:
            return cls._loggers[logger_key]

        logger = logging.getLogger(logger_key)
        logger.setLevel(level.upper())
        logger.propagate = False  # Prevent duplicate logs

        # Resolve log file
        if not log_file and task_paths:
            log_file = task_paths.get_log_path(run_id=run_id)

        # Determine log directory
        log_dir = os.path.dirname(log_file) if log_file else cls._default_log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Finalize log_file if still unset
        if not log_file:
            log_file = os.path.join(log_dir, f"{name}.log")

        file_handler = cls._setup_file_handler(log_file, level, use_json)
        console_handler = cls._setup_console_handler(level, use_color)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        cls._loggers[logger_key] = logger
        return logger

    @staticmethod
    def _setup_file_handler(
        filepath: str, level: str, use_json: bool
    ) -> logging.Handler:
        """
        Creates and configures a file handler for logging.

        Args:
            filepath (str): Path to the log file.
            level (str): Logging level threshold.
            use_json (bool): Whether to use JSON formatting.

        Returns:
            logging.Handler: A file handler with formatter attached.
        """
        handler = logging.FileHandler(filepath, encoding="utf-8")
        handler.setLevel(level.upper())
        formatter = LoggerManager._get_formatter(use_json=use_json, color=False)
        handler.setFormatter(formatter)
        return handler

    @staticmethod
    def _setup_console_handler(level: str, use_color: bool) -> logging.Handler:
        """
        Creates and configures a console (stdout) handler.

        Args:
            level (str): Logging level threshold.
            use_color (bool): Whether to use colored output (requires `colorlog`).

        Returns:
            logging.Handler: A stream handler with formatter attached.
        """
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level.upper())
        formatter = LoggerManager._get_formatter(use_json=False, color=use_color)
        handler.setFormatter(formatter)
        return handler

    @staticmethod
    def _get_formatter(
        use_json: bool = False, color: bool = False
    ) -> logging.Formatter:
        """
        Returns a log formatter object based on configuration.

        Args:
            use_json (bool): If True, returns a JSON formatter (for structured logs).
            color (bool): If True and colorlog is installed, returns a colored
                formatter.

        Returns:
            logging.Formatter: A formatter instance.
        """
        if use_json:
            return JsonLogFormatter()

        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        datefmt = "%Y-%m-%d %H:%M:%S"

        if color and COLORLOG_AVAILABLE:
            return ColoredFormatter(
                fmt="%(log_color)s" + fmt,
                datefmt=datefmt,
                log_colors={
                    'DEBUG': 'cyan',
                    'INFO': 'green',
                    'WARNING': 'yellow',
                    'ERROR': 'red',
                    'CRITICAL': 'bold_red',
                },
            )
        else:
            return logging.Formatter(fmt, datefmt)


class JsonLogFormatter(logging.Formatter):
    """
    A custom formatter that outputs logs in JSON format.

    Designed for use in file handlers when structured logging is needed
    (e.g., for log ingestion pipelines, testing, or analytics).

    Example Output:
        {
            "timestamp": "2025-05-07 13:12:01",
            "level": "INFO",
            "logger": "chunker",
            "message": "Chunking started",
            "email_id": "abc123"
        }

    Supports extra data via `extra={"extra_data": {...}}` in logging calls.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "timestamp": self.formatTime(record, "%Y-%m-%d %H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_record.update(record.extra_data)
        return json.dumps(log_record)

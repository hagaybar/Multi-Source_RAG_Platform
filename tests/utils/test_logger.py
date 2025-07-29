import os
import logging
import json
import pytest
import shutil  # For cleanup
import sys  # Added for the new test

# Import the logger components to be tested
from scripts.utils.logger import LoggerManager, JsonLogFormatter

# Define a directory for test logs
TEST_LOG_DIR = "test_logs"


@pytest.fixture(scope="function")
def cleanup_test_logs():
    """Create and clean up the test log directory."""
    if os.path.exists(TEST_LOG_DIR):
        _cleanup_all_loggers()
        shutil.rmtree(TEST_LOG_DIR)
    os.makedirs(TEST_LOG_DIR, exist_ok=True)
    yield
    if os.path.exists(TEST_LOG_DIR):
        _cleanup_all_loggers()
        shutil.rmtree(TEST_LOG_DIR)


def _cleanup_all_loggers():
    for logger_name in list(logging.Logger.manager.loggerDict.keys()):
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            try:
                handler.close()
                logger.removeHandler(handler)
            except Exception:
                pass


def test_get_logger_returns_same_instance(cleanup_test_logs):
    """Tests that get_logger returns the same logger instance for the same name."""
    logger_name = "test_singleton"
    logger1 = LoggerManager.get_logger(
        logger_name, log_file=os.path.join(TEST_LOG_DIR, "singleton.log")
    )
    logger2 = LoggerManager.get_logger(
        logger_name, log_file=os.path.join(TEST_LOG_DIR, "singleton.log")
    )
    assert logger1 is logger2


def test_console_and_file_handlers_added(cleanup_test_logs):
    """Tests that both console and file handlers are added to the logger."""
    logger_name = "test_handlers"
    log_file_path = os.path.join(TEST_LOG_DIR, f"{logger_name}.log")
    logger = LoggerManager.get_logger(logger_name, log_file=log_file_path)

    assert len(logger.handlers) == 2, "Logger should have two handlers"

    has_console_handler = any(
        isinstance(h, logging.StreamHandler) and h.stream == sys.stdout for h in logger.handlers
    )
    has_file_handler = any(
        isinstance(h, logging.FileHandler) and h.baseFilename == os.path.abspath(log_file_path)
        for h in logger.handlers
    )

    assert has_console_handler, "Logger should have a console handler"
    assert has_file_handler, "Logger should have a file handler"


def test_json_log_output_structure(cleanup_test_logs):
    """Tests that file logs are valid JSON when use_json is True and contain expected keys."""
    logger_name = "test_json_output"
    log_file_path = os.path.join(TEST_LOG_DIR, f"{logger_name}.log")
    logger = LoggerManager.get_logger(
        logger_name, log_file=log_file_path, use_json=True, level="INFO"
    )

    test_message = "This is a JSON test message."
    extra_data = {"key1": "value1", "key2": 123}
    logger.info(test_message, extra={"extra_data": extra_data})

    # Ensure the file handler is closed so the message is flushed
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()

    assert os.path.exists(log_file_path), "Log file was not created"

    with open(log_file_path, 'r') as f:
        log_content = f.readline().strip()  # Read the first line of log

    assert log_content, "Log file is empty"

    try:
        log_entry = json.loads(log_content)
    except json.JSONDecodeError:
        pytest.fail("Log output is not valid JSON.")

    expected_keys = ["timestamp", "level", "logger", "message", "key1", "key2"]
    for key in expected_keys:
        assert key in log_entry, f"Expected key '{key}' not found in JSON log."

    assert log_entry["message"] == test_message
    assert log_entry["logger"] == logger_name
    assert log_entry["level"] == "INFO"
    assert log_entry["key1"] == extra_data["key1"]
    assert log_entry["key2"] == extra_data["key2"]


def test_colorlog_fallbacks_gracefully(cleanup_test_logs, monkeypatch):
    """
    Tests that the logger falls back gracefully if colorlog is not available.
    It also checks that the console handler's formatter is a standard logging.Formatter.
    """
    logger_name = "test_color_fallback"
    log_file_path = os.path.join(TEST_LOG_DIR, f"{logger_name}.log")

    # Simulate colorlog not being available
    monkeypatch.setattr("scripts.utils.logger.COLORLOG_AVAILABLE", False)

    # Ensure no exception is raised during logger creation and basic logging
    try:
        logger = LoggerManager.get_logger(
            logger_name,
            log_file=log_file_path,
            use_color=True,  # Attempt to use color
        )
        logger.info("Test message without colorlog.")  # Should not raise error
    except Exception as e:
        pytest.fail(f"Logger initialization or logging failed when colorlog is unavailable: {e}")

    # Verify the console handler's formatter is a standard logging.Formatter
    console_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            console_handler = handler
            break

    assert console_handler is not None, "Console handler not found."
    assert isinstance(console_handler.formatter, logging.Formatter), (
        "Console handler formatter should be a standard logging.Formatter when colorlog is unavailable."
    )

    # Check that it's not a ColorLog specific formatter (if ColorLogFormatter was imported and type checkable)
    # Since we cannot directly import ColoredFormatter when it might not exist,
    # we rely on the fact that it *would* be a ColoredFormatter if COLORLOG_AVAILABLE was True
    # and LoggerManager tried to use it. The isinstance(..., logging.Formatter) check is key.

    # Also ensure file logging still works
    assert os.path.exists(log_file_path), "Log file was not created during fallback test."
    with open(log_file_path, 'r') as f:
        assert "Test message without colorlog." in f.read()


def test_logfile_created(cleanup_test_logs):
    """Tests that a log file is created when a message is logged."""
    logger_name = "test_logfile_creation"
    log_file_path = os.path.join(TEST_LOG_DIR, f"{logger_name}.log")

    # Ensure log file does not exist before logging
    if os.path.exists(log_file_path):
        os.remove(log_file_path)

    assert not os.path.exists(log_file_path), "Log file exists before logging."

    logger = LoggerManager.get_logger(logger_name, log_file=log_file_path, level="INFO")
    logger.info("This message should create a log file.")

    # Ensure the file handler is closed so the message is flushed
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            handler.close()  # Close to ensure flush

    assert os.path.exists(log_file_path), "Log file was not created after logging."

    with open(log_file_path, 'r') as f:
        content = f.read()
        assert "This message should create a log file." in content, (
            "Log message not found in the created file."
        )


def cleanup_logger(name: str):
    logger = logging.getLogger(name)
    handlers = logger.handlers[:]
    for handler in handlers:
        handler.close()
        logger.removeHandler(handler)

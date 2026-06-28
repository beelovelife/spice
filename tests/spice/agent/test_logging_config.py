from __future__ import annotations

import logging

import spice.agent.logging_config as logging_config
import spice.cli.main as cli_main
from typer.testing import CliRunner


def test_configure_logging_writes_to_explicit_path(tmp_path, monkeypatch) -> None:
    logger = logging.getLogger("spice")
    old_handlers = list(logger.handlers)
    old_path = logging_config._LOG_PATH
    for handler in old_handlers:
        logger.removeHandler(handler)
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setattr(logging_config, "_LOG_PATH", None)

    path = tmp_path / "spice.log"
    try:
        configured_path = logging_config.configure_logging(log_path=path)
        logging_config.get_logger("spice.test").info("hello logging")
        for handler in logging.getLogger("spice").handlers:
            handler.flush()

        assert configured_path == path
        assert path.exists()
        assert "hello logging" in path.read_text(encoding="utf-8")
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        monkeypatch.setattr(logging_config, "_CONFIGURED", bool(old_handlers))
        monkeypatch.setattr(logging_config, "_LOG_PATH", old_path)


def test_get_logger_prefixes_non_spice_names(monkeypatch) -> None:
    assert logging_config.get_logger("agent.loop").name == "spice.agent.loop"
    assert logging_config.get_logger("spice.agent.loop").name == "spice.agent.loop"


def test_get_logger_does_not_configure_logging(monkeypatch) -> None:
    logger = logging.getLogger("spice")
    old_handlers = list(logger.handlers)
    old_path = logging_config._LOG_PATH
    for handler in old_handlers:
        logger.removeHandler(handler)
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setattr(logging_config, "_LOG_PATH", None)

    try:
        logging_config.get_logger("spice.agent.loop")

        assert logging_config._CONFIGURED is False
        assert logger.handlers == []
        assert logging_config.log_path() == logging_config.LOG_PATH
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        monkeypatch.setattr(logging_config, "_CONFIGURED", bool(old_handlers))
        monkeypatch.setattr(logging_config, "_LOG_PATH", old_path)


def test_explicit_debug_configuration_after_get_logger(tmp_path, monkeypatch) -> None:
    logger = logging.getLogger("spice")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    old_path = logging_config._LOG_PATH
    for handler in old_handlers:
        logger.removeHandler(handler)
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setattr(logging_config, "_LOG_PATH", None)

    path = tmp_path / "spice.debug.log"
    try:
        child = logging_config.get_logger("agent.loop")
        configured_path = logging_config.configure_logging(debug=True, log_path=path)
        child.debug("debug is enabled")
        for handler in logger.handlers:
            handler.flush()

        assert configured_path == path
        assert logging_config.log_path() == path
        assert logger.level == logging.DEBUG
        assert "debug is enabled" in path.read_text(encoding="utf-8")
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        monkeypatch.setattr(logging_config, "_CONFIGURED", bool(old_handlers))
        monkeypatch.setattr(logging_config, "_LOG_PATH", old_path)


def test_cli_debug_option_configures_logging_before_command(tmp_path, monkeypatch) -> None:
    logger = logging.getLogger("spice")
    old_handlers = list(logger.handlers)
    old_level = logger.level
    old_propagate = logger.propagate
    old_path = logging_config._LOG_PATH
    for handler in old_handlers:
        logger.removeHandler(handler)
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    monkeypatch.setattr(logging_config, "_LOG_PATH", None)
    debug_path = tmp_path / "cli.debug.log"
    monkeypatch.setattr(logging_config, "DEBUG_LOG_PATH", debug_path)
    monkeypatch.setattr(cli_main, "set_process_title", lambda: None)

    try:
        result = CliRunner().invoke(cli_main.app, ["--debug", "--version"])

        assert result.exit_code == 0
        assert logging_config.log_path() == debug_path
        assert logger.level == logging.DEBUG
    finally:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        for handler in old_handlers:
            logger.addHandler(handler)
        logger.setLevel(old_level)
        logger.propagate = old_propagate
        monkeypatch.setattr(logging_config, "_CONFIGURED", bool(old_handlers))
        monkeypatch.setattr(logging_config, "_LOG_PATH", old_path)

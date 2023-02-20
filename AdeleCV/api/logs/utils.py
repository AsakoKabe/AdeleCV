import logging

from config import get_settings


def enable_logs(
        handler,
        name=get_settings().LOGGER_NAME,
        formatter=logging.Formatter('%(levelname)s - %(message)s'),
        level=logging.DEBUG,
) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(
        name=get_settings().LOGGER_NAME
) -> logging.Logger:
    return logging.getLogger(name)

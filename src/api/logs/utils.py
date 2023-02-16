import logging

from config import get_settings


def enable_logs(
        handler,
        name=get_settings().LOGGER_NAME,
        formatter=logging.Formatter('%(levelname)s - %(message)s')
) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(
        name=get_settings().LOGGER_NAME
) -> logging.Logger:
    return logging.getLogger(name)

import logging

from adelecv.api.config import Settings


def enable_logs(
        handler: logging.Handler,
        name: str = Settings.LOGGER_NAME,
        formatter: logging.Formatter = logging.Formatter(
            '%(levelname)s - %(message)s'
        ),
        level: int = logging.DEBUG,
) -> None:
    """
    Enabling python logging.

    :param handler: logging.Handler for collecting logging
     (example StreamHandler)
    :param name: name logger
    :param formatter: str format python logging.
     default - %(levelname)s - %(message)s
    :param level: python logging level
    """

    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler.setLevel(level)
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def get_logger(
        name: str = Settings.LOGGER_NAME
) -> logging.Logger:
    """
    Get logger by name

    :param name: python logging logger name
    :return: python logging logger
    """
    return logging.getLogger(name)

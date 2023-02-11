import logging

from config import get_settings


def enable_logs(handle_type):
    # todo: другие handles file stream
    logger = logging.getLogger(get_settings().LOGGER_NAME)
    logger.setLevel(logging.DEBUG)
    handle = handle_type()
    handle.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(levelname)s - %(message)s')
    # add formatter to ch
    handle.setFormatter(formatter)
    logger.addHandler(handle)


def get_logger():
    return logging.getLogger(get_settings().LOGGER_NAME)

import logging
import sys


class LogConsoleHandler(logging.StreamHandler):
    """ Class to redistribute python logging data """

    def __init__(self, **kwargs):
        # Initialize the Handler
        super().__init__(stream=sys.stdout)

        # optional take format
        # setFormatter function is derived from logging.Handler
        for key, value in kwargs.items():
            if f"{key}" == "format":
                self.setFormatter(value)

        self._logs = []

    def emit(self, record: logging.LogRecord) -> None:
        """ Overload of logging.Handler method """
        record = self.format(record)
        self._logs.append(record)

    @property
    def logs(self):
        return self._logs

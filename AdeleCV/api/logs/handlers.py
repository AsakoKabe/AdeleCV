import logging


class LogMonitoringHandler(logging.Handler):
    """ Class to redistribute python logging data """

    def __init__(self, *args, **kwargs):
        # Initialize the Handler
        logging.Handler.__init__(self, *args)

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

    def pop_logs(self) -> list[str]:
        logs = self._logs.copy()
        self._logs.clear()

        return logs

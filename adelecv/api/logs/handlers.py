import logging


class LogMonitoringHandler(logging.Handler):
    """
    :meta private:

    Handler class for collecting internal logs.
    Used for notifications in the UI
    """

    def __init__(self, *args, **kwargs):
        logging.Handler.__init__(self, *args)
        for key, value in kwargs.items():
            if f"{key}" == "format":
                self.setFormatter(value)

        self._logs = []

    def emit(self, record: logging.LogRecord) -> None:
        """
        :meta private:

        Overload of logging.Handler method

        :param record: logging.LogRecord
        """
        record = self.format(record)
        self._logs.append(record)

    def pop_logs(self) -> list[str]:
        """
        :meta private:
        :return:
        """
        logs = self._logs.copy()
        self._logs.clear()

        return logs

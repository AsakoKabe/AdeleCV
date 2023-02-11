import logging


class LogMonitoringHandler(logging.Handler):
    """ Class to redistribute python logging data """

    # have a class member to store the existing logger
    # logger_instance = logging.getLogger("test")

    def __init__(self, *args, **kwargs):
        # Initialize the Handler
        logging.Handler.__init__(self, *args)

        # optional take format
        # setFormatter function is derived from logging.Handler
        for key, value in kwargs.items():
            if "{}".format(key) == "format":
                self.setFormatter(value)

        # make the logger send data to this class
        # self.logger_instance.addHandler(self)
        self._logs = []

    def emit(self, record):
        """ Overload of logging.Handler method """
        record = self.format(record)
        self._logs.append(record)

    def pop_logs(self):
        logs = self._logs.copy()
        self._logs.clear()

        return logs

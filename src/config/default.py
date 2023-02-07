import os
from os import environ
from pathlib import Path


class DefaultSettings:
    """
    Default configs for application.
    """

    TMP_PATH: Path = Path(os.getenv('TMP_PATH'))
    LOGS_PATH: Path = TMP_PATH / 'logs'
    WEIGHTS_PATH: Path = TMP_PATH / 'weights'
    CACHE_PATH: Path = TMP_PATH / 'cache'

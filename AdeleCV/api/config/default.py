import dataclasses
from pathlib import Path
from uuid import uuid4


@dataclasses.dataclass
class Settings:
    """
    Default configs for application.

    """

    TMP_PATH: Path = Path('./tmp') / uuid4().hex
    TENSORBOARD_LOGS_PATH: Path = TMP_PATH / 'tensorboard'
    WEIGHTS_PATH: Path = TMP_PATH / 'weights'
    LOGGER_NAME: str = 'autodl'

    @staticmethod
    def update_tmp_path(new_tmp_path: Path):
        Settings.TMP_PATH = new_tmp_path
        Settings.TENSORBOARD_LOGS_PATH = Settings.TMP_PATH / 'tensorboard'
        Settings.WEIGHTS_PATH = Settings.TMP_PATH / 'weights'

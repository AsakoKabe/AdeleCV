import dataclasses
from pathlib import Path
from uuid import uuid4


@dataclasses.dataclass
class Settings:
    """
    Default configs for application.

    :param TMP_PATH: Path to save logs and weights
    :param TENSORBOARD_LOGS_PATH: Path to tensorboard logs
    :param WEIGHTS_PATH: Path to weight models
    :param LOGGER_NAME: Name logging logger
    :param DASHBOARD_PORT: Application port
    :param FIFTYONE_PORT: Fiftyone app port
    :param TENSORBOARD_PORT: Tensorboard app port
    """

    TMP_PATH: Path = Path('./tmp') / uuid4().hex
    TENSORBOARD_LOGS_PATH: Path = TMP_PATH / 'tensorboard'
    WEIGHTS_PATH: Path = TMP_PATH / 'weights'
    LOGGER_NAME: str = 'AdeleCV'
    DASHBOARD_PORT: int = 8080
    FIFTYONE_PORT: int = 5151
    TENSORBOARD_PORT: int = 6006

    @staticmethod
    def update_tmp_path(new_tmp_path: Path) -> None:
        Settings.TMP_PATH = new_tmp_path
        Settings.TENSORBOARD_LOGS_PATH = Settings.TMP_PATH / 'tensorboard'
        Settings.WEIGHTS_PATH = Settings.TMP_PATH / 'weights'

    @staticmethod
    def update_dashboard_port(port: int) -> None:
        Settings.DASHBOARD_PORT = port

    @staticmethod
    def update_fiftyone_port(port: int) -> None:
        Settings.FIFTYONE_PORT = port

    @staticmethod
    def update_tensorboard_port(port: int) -> None:
        Settings.TENSORBOARD_PORT = port

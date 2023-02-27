from abc import ABC
from os import environ
from pathlib import Path
from uuid import uuid4

import pandas as pd
import fiftyone as fo
from tensorboard import program
from dotenv import load_dotenv

from api.config import Settings
from api.logs import get_logger
from api.models.segmentations import SegmentationModel


def _load_settings_from_env() -> None:
    load_dotenv(dotenv_path='../.env')
    if environ.get('TMP_PATH') is not None:
        Settings.update_tmp_path(Path(environ.get('TMP_PATH')) / uuid4().hex)


class BaseTask(ABC):
    def __init__(self):
        _load_settings_from_env()
        self.models: list[SegmentationModel] = []
        self._stats_models = pd.DataFrame()
        self._weights_dir = Settings.WEIGHTS_PATH
        self._hp_optimizer = None
        self._dataset = None
        self._session_dataset = fo.launch_app(remote=True)
        self._tb = program.TensorBoard()
        self._tb.configure(argv=[None, '--logdir', Settings.TENSORBOARD_LOGS_PATH.as_posix()])
        self._tb.launch()

    def _create_dataset_session(self) -> None:
        get_logger().debug("Create fifty one dataset session")
        self._session_dataset.dataset = self._dataset.fo_dataset

    def _run_optimize(self) -> None:
        self._hp_optimizer.optimize()

    @property
    def stats_models(self) -> pd.DataFrame:
        return self._stats_models


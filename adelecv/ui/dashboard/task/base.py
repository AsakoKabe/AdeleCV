from __future__ import annotations

from abc import ABC
from os import environ
from pathlib import Path
from uuid import uuid4

import fiftyone as fo
import pandas as pd
from dotenv import load_dotenv
from tensorboard import program

from adelecv.api.config import Settings
from adelecv.api.logs import get_logger
from adelecv.api.models.segmentations import SegmentationModel


def _load_settings_from_env(env_path: Path) -> None:
    load_dotenv(dotenv_path=env_path)

    tmp_path = environ.get('TMP_PATH')
    dashboard_port = environ.get('DASHBOARD_PORT')
    fiftyone_port = environ.get('FIFTYONE_PORT')
    tensorboard_port = environ.get('TENSORBOARD_PORT')

    if tmp_path is not None:
        Settings.update_tmp_path(Path(tmp_path) / uuid4().hex)
    if dashboard_port is not None:
        Settings.update_dashboard_port(int(dashboard_port))
    if fiftyone_port is not None:
        Settings.update_fiftyone_port(int(fiftyone_port))
    if tensorboard_port is not None:
        Settings.update_tensorboard_port(int(tensorboard_port))


class BaseTask(ABC):
    def __init__(self):
        self._weights_dir = None
        self.models: list[SegmentationModel] = []
        self._stats_models = pd.DataFrame()
        self._hp_optimizer = None
        self._dataset = None
        self._tb = None
        self._session_dataset = None
        self._img_shape = None

    def launch(self, env_path: Path | None):
        if env_path is not None:
            _load_settings_from_env(Path(env_path))

        self._weights_dir = Settings.WEIGHTS_PATH
        self._session_dataset = fo.launch_app(
            remote=True, port=Settings.FIFTYONE_PORT
        )
        self._tb = program.TensorBoard()
        self._tb.configure(
            argv=[
                None,
                '--logdir', Settings.TENSORBOARD_LOGS_PATH.as_posix(),
                '--port', str(Settings.TENSORBOARD_PORT)
            ]
        )
        self._tb.launch()

    def _create_dataset_session(self) -> None:
        get_logger().debug("Create fifty one dataset session")
        self._session_dataset.dataset = self._dataset.fo_dataset

    def _run_optimize(self) -> None:
        self._hp_optimizer.optimize()

    @property
    def stats_models(self) -> pd.DataFrame:
        return self._stats_models

    @property
    def img_shape(self) -> list[int, int]:
        return self._img_shape

    @img_shape.setter
    def img_shape(self, new_val: list[int, int]):
        self._img_shape = new_val

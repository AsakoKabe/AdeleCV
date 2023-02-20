from abc import ABC

import pandas as pd
import fiftyone as fo
from tensorboard import program

from api.logs import get_logger
from api.models.segmentations import SegmentationModel
from config import get_settings


class BaseTask(ABC):
    def __init__(self):
        self.models: list[SegmentationModel] = []
        self._stats_models = pd.DataFrame()
        self._weights_dir = get_settings().WEIGHTS_PATH
        self._hp_optimizer = None
        self._dataset = None
        self._session_dataset = fo.launch_app(remote=True)
        self._tb = program.TensorBoard()
        self._tb.configure(argv=[None, '--logdir', get_settings().TENSORBOARD_LOGS_PATH.as_posix()])
        self._tb.launch()

    def _create_dataset_session(self) -> None:
        get_logger().debug("Create fifty one dataset session")
        self._session_dataset.dataset = self._dataset.fo_dataset

    def _run_optimize(self) -> None:
        self._hp_optimizer.optimize()

    @property
    def stats_models(self) -> pd.DataFrame:
        return self._stats_models

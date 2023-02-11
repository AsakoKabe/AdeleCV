import zipfile
from abc import abstractmethod, ABC
from typing import List

import pandas as pd
import fiftyone as fo
from tensorboard import program

from api.models.segmentations import SegmentationModel
from config import get_settings


class BaseTask(ABC):
    def __init__(self):
        self.models: List[SegmentationModel] = []
        self._stats_models = pd.DataFrame()
        self._weights_dir = get_settings().WEIGHTS_PATH
        self._hp_optimizer = None
        self._dataset = None
        self._session_dataset = fo.launch_app(remote=True)
        self._tb = program.TensorBoard()
        self._tb.configure(argv=[None, '--logdir', get_settings().TENSORBOARD_LOGS_PATH.as_posix()])
        self._tb.launch()

    def export_weights(self, id_selected):
        zip_path = self._weights_dir.parent / 'weights.zip'
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in self._weights_dir.rglob("*"):
                if entry.stem in id_selected:
                    zip_file.write(entry, entry.relative_to(self._weights_dir))

        return zip_path

    def _create_dataset_session(self):
        self._session_dataset.dataset = self._dataset.fo_dataset

    def _run_optimize(self):
        self._hp_optimizer.optimize()

    @property
    def stats_models(self):
        return self._stats_models

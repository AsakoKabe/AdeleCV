import itertools
import os
import zipfile
from pathlib import Path
from typing import Tuple, Any, List

import pandas as pd
import fiftyone as fo
from tensorboard import program

from data.dataset.SegmentationDataset import \
    SegmentationDataset
from models.semantic.SegmentationModel import SegmentationModel
from optimize.hp_optimizer import HPOptimizer
from task.base import BaseTask


class SegmentationTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.models: List[SegmentationModel] = []
        self._stats_models = pd.DataFrame()
        self._weights_dir = Path(os.getenv('TMP_PATH')) / 'weights'
        self.hp_optimizer = None
        self.dataset = None
        self.session_dataset = fo.launch_app(remote=True)
        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', f'{os.getenv("TMP_PATH")}/logs'])
        self.tb.launch()

    def add_stats_model(self, model: SegmentationModel):
        self._stats_models = pd.concat(
            [self._stats_models, pd.DataFrame([model.stats_model])],
            ignore_index=True
        )

    @property
    def stats_models(self):
        return self._stats_models

    def create_optimizer(self, params):
        self.hp_optimizer = HPOptimizer(
            params["architectures"],
            params["lr_range"],
            params["optimizers"],
            params["loss_fns"],
            params["epoch_range"],
            params["strategy"],
            params["num_trials"],
            params["device"],
            # todo: переписать, добавление информации о модели в таск
            self
        )

    def run_optimize(self):
        self.hp_optimizer.optimize()

    def run(self):
        self.hp_optimizer.optimize()

    def load_dataset(
            self,
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size,
    ):
        self.dataset = SegmentationDataset(
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size
        )

    def create_dataset_session(self):
        self.session_dataset.dataset = self.dataset.fo_dataset

    def export_weights(self, id_selected):
        zip_path = self._weights_dir.parent / 'weights.zip'
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in self._weights_dir.rglob("*"):
                if entry.stem in id_selected:
                    zip_file.write(entry, entry.relative_to(self._weights_dir))

        return zip_path

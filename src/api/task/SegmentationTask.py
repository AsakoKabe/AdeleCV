import os
import zipfile
from typing import List

import pandas as pd
import fiftyone as fo
from tensorboard import program

from config import get_settings
from api.data.segmentations import SegmentationDataset
from api.models.semantic import SegmentationModel
from api.optimize import HPOptimizer
from . import BaseTask
from ..logs import get_logger


class SegmentationTask(BaseTask):
    def __init__(self):
        super().__init__()
        self.models: List[SegmentationModel] = []
        self._stats_models = pd.DataFrame()
        self._weights_dir = get_settings().WEIGHTS_PATH
        self.hp_optimizer = None
        self.dataset = None
        self.session_dataset = fo.launch_app(remote=True)
        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', get_settings().TENSORBOARD_LOGS_PATH.as_posix()])
        self.tb.launch()
        # os.mkdir(get_settings().TMP_PATH)

    @property
    def stats_models(self):
        return self._stats_models

    def train(self, params):
        logger = get_logger()
        logger.info("Train models started")
        self.hp_optimizer = HPOptimizer(
            params["architectures"],
            params['encoders'],
            params["lr_range"],
            params["optimizers"],
            params["loss_fns"],
            params["epoch_range"],
            params["strategy"],
            params["num_trials"],
            params["device"],
            dataset=self.dataset,
        )
        self._run_optimize()
        self._stats_models = self.hp_optimizer.stats_models
        logger.info("Train models is over")

    def _run_optimize(self):
        self.hp_optimizer.optimize()

    def load_dataset(
            self,
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size,
    ):
        logger = get_logger()
        logger.info("Creating a dataset")
        self.dataset = SegmentationDataset(
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size
        )
        self._create_dataset_session()
        logger.info("Dataset created")

    def _create_dataset_session(self):
        self.session_dataset.dataset = self.dataset.fo_dataset

    def export_weights(self, id_selected):
        zip_path = self._weights_dir.parent / 'weights.zip'
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in self._weights_dir.rglob("*"):
                if entry.stem in id_selected:
                    zip_file.write(entry, entry.relative_to(self._weights_dir))

        return zip_path

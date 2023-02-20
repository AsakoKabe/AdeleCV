from __future__ import annotations

from api.data.segmentations import SegmentationDataset
from api.optimize import HPOptimizer, HyperParamsSegmentation
from api.data.segmentations.types import DatasetType
from .base import BaseTask


class SegmentationTask(BaseTask):
    # def __init__(self):
    #     super().__init__()

    def train(
            self,
            params: dict[
                str,
                int | float | str | list | tuple
            ]
    ) -> None:
        self._hp_optimizer = HPOptimizer(
            hyper_params=HyperParamsSegmentation(
                architectures=params["architectures"],
                encoders=params['encoders'],
                pretrained_weights=params['pretrained_weight'],
                lr_range=params["lr_range"],
                optimizers=params["optimizers"],
                loss_fns=params["loss_fns"],
                epoch_range=params["epoch_range"],
                optimize_score=params['optimize_score'],
                strategy=params["strategy"],
            ),
            num_trials=params["num_trials"],
            device=params["device"],
            dataset=self._dataset,
        )
        self._run_optimize()
        self._stats_models = self._hp_optimizer.stats_models

    def load_dataset(
            self,
            dataset_path: str,
            dataset_type: DatasetType,
            img_size: tuple[int, int],  # height, width
            split: tuple[float, float, float],
            batch_size: int,
    ) -> None:
        self._dataset = SegmentationDataset(
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size
        )
        self._create_dataset_session()

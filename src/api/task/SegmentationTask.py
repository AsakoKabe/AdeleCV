from api.data.segmentations import SegmentationDataset
from api.optimize import HPOptimizer
from api.logs import get_logger
from .base import BaseTask


class SegmentationTask(BaseTask):
    def __init__(self):
        super().__init__()

    def train(self, params):
        logger = get_logger()
        logger.info("Train models started")
        self._hp_optimizer = HPOptimizer(
            params["architectures"],
            params['encoders'],
            params['pretrained-weight'],
            params["lr_range"],
            params["optimizers"],
            params["loss_fns"],
            params["epoch_range"],
            params["strategy"],
            params["num_trials"],
            params["device"],
            dataset=self._dataset,
        )
        self._run_optimize()
        self._stats_models = self._hp_optimizer.stats_models
        logger.info("Train models is over")

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
        self._dataset = SegmentationDataset(
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size
        )
        self._create_dataset_session()
        logger.info("Dataset created")

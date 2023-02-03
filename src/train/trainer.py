import os

import torch
import fiftyone as fo
from tensorboard import program


from data.dataset.SegmentationDataset import SegmentationDataset
from optimize.hp_optimizer import HPOptimizer
from task.SegmentationTask import SegmentationTask
from task.base import BaseTask


class Trainer:
    def __init__(self, task: BaseTask = None, cuda: bool = True):
        self.hp_optimizer = None
        self.task = None
        self.dataset = None
        self.session_dataset = fo.launch_app(remote=True)
        self.tb = program.TensorBoard()
        self.tb.configure(argv=[None, '--logdir', f'{os.getenv("TMP_PATH")}/logs'])
        self.tb.launch()

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

    def create_task(self):
        self.task = SegmentationTask(self.dataset)

    def create_optimizer(self, params):
        if not self.task:
            raise AttributeError('Task not created')

        self.hp_optimizer = HPOptimizer(
            params["architectures"],
            params["lr_range"],
            params["optimizers"],
            params["loss_fns"],
            params["epoch_range"],
            params["strategy"],
            params["num_trials"],
            params["device"],
            self.task
        )

    def run_optimize(self):
        self.hp_optimizer.optimize()


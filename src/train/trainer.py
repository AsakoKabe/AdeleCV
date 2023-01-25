import torch
import fiftyone as fo

from data.dataset.SemanticSegmentationDataset import SemanticDataset
from optimize.hp_optimizer import HPOptimizer
from task.SemanticSegmentationTask import SemanticSegmentationTask
from task.base import BaseTask


class Trainer:
    def __init__(self, task: BaseTask = None, cuda: bool = True):
        self.hp_optimizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.task = None
        self.dataset = None
        self.session_dataset = fo.launch_app(remote=True)
        # self.session_dataset.wait()

    def run(self):
        # self.task.fit_models()
        self.hp_optimizer.optimize()

    def load_dataset(
            self,
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size,
    ):
        self.dataset = SemanticDataset(
            dataset_path,
            dataset_type,
            img_size,
            split,
            batch_size
        )

    def create_dataset_session(self):
        self.session_dataset.dataset = self.dataset.fo_dataset

    def create_task(self):
        self.task = SemanticSegmentationTask(self.dataset)
        self.task.device = self.device
    
    def create_optimizer(self):
        if not self.task:
            raise AttributeError('Task not created')

        self.hp_optimizer = HPOptimizer(self.task)


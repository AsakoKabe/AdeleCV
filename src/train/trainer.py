import torch

from .task.base import BaseTask


class Trainer:
    def __init__(self, task: BaseTask, cuda: bool = True):
        self.device = torch.device('cuda' if torch.cuda.is_available() and cuda else 'cpu')
        self.task = task
        self.task.device = self.device

    def run(self):
        self.task.fit_models()

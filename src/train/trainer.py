from train.task.base import BaseTask


class Trainer:
    def __init__(self, task: BaseTask):
        self.task = task

    def run(self):
        self.task.fit_models()

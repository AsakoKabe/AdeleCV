from data.dataset.base import BaseDataset


class BaseTask:
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

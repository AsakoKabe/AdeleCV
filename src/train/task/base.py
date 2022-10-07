from abc import abstractmethod, ABC

from data.dataset.base import BaseDataset


class BaseTask(ABC):
    def __init__(self, dataset: BaseDataset):
        self.dataset = dataset

    @abstractmethod
    def fit_models(self):
        pass

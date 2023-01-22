import itertools
from typing import Tuple, Any, List

from torch.optim import AdamW, Adagrad, SGD
from torch.utils.data import DataLoader

from data.dataset.SemanticSegmentationDataset import \
    SemanticDataset
from models.semantic.SemanticModel import SemanticModel
from models.semantic.DeepLabV3MobileNet import DeepLabV3MobileNet
from models.semantic.LRASPPMobileNetV3 import LRASPPMobileNetV3
from task.base import BaseTask


class SemanticSegmentationTask(BaseTask):
    def __init__(self, dataset: SemanticDataset):
        super().__init__(
            dataset=dataset
        )
        self.__optimizers = [AdamW, Adagrad, SGD]
        self.__lr = [0.001]
        self.__loss_fns = []

        self.models: List[SemanticModel] = []






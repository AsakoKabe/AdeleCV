import itertools

from torch.optim import AdamW, Adagrad, SGD

from models.semantic.DeepLabV3MobileNet import DeepLabV3MobileNet
from models.semantic.LRASPPMobileNetV3 import LRASPPMobileNetV3
from train.task.base import BaseTask
from train.task.loss import bce_loss, dice_loss


class SemanticSegmentationTask(BaseTask):
    def __init__(self, dataset):
        super(SemanticSegmentationTask, self).__init__(
            dataset=dataset
        )
        self.__optimizers = [AdamW, Adagrad, SGD]
        self.__lr = [0.001]
        self.__loss_fns = [bce_loss, dice_loss]

        self.models = []

    def __generate_models(self):
        combinations_params = list(
            itertools.product(
                self.__optimizers,
                self.__lr,
                self.__loss_fns)
        )
        for params in combinations_params:
            self.models.append(
                DeepLabV3MobileNet(
                    optimizer=params[0],
                    loss_fn=params[1],
                    lr=params[2]
                )
            )
            self.models.append(
                LRASPPMobileNetV3(
                    optimizer=params[0],
                    loss_fn=params[1],
                    lr=params[2]
                )
            )

    def __fit_models(self):
        pass

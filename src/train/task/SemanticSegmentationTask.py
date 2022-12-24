import itertools
from typing import Tuple, Any, List

from torch.optim import AdamW, Adagrad, SGD
from torch.utils.data import DataLoader

from data.dataset.SemanticSegmentationDataset import \
    SemanticSegmentationDataset
from models.semantic.BaseSemanticModel import BaseSemanticModel
from models.semantic.DeepLabV3MobileNet import DeepLabV3MobileNet
from models.semantic.LRASPPMobileNetV3 import LRASPPMobileNetV3
from train.task.base import BaseTask
from train.task.loss import bce_loss, dice_loss


class SemanticSegmentationTask(BaseTask):
    def __init__(self, dataset: SemanticSegmentationDataset):
        super().__init__(
            dataset=dataset
        )
        self.__optimizers = [AdamW, Adagrad, SGD]
        self.__lr = [0.001]
        self.__loss_fns = [bce_loss, dice_loss]

        self.models: List[BaseSemanticModel] = []
        self.__generate_models()

    def __generate_models(self):
        combinations_params = list(
            itertools.product(
                self.__optimizers,
                self.__loss_fns,
                self.__lr,
                )
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

    def _create_dataloaders(self) -> Tuple[
        DataLoader[Any],
        DataLoader[Any],
        DataLoader[Any]
    ]:

        train_dataloader = DataLoader(
            self.dataset.train_dataset,
            batch_size=self.dataset.batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            self.dataset.val_dataset,
            batch_size=self.dataset.batch_size,
            shuffle=True
        )
        test_dataloader = DataLoader(
            self.dataset.test_dataset,
            batch_size=self.dataset.batch_size,
            shuffle=True
        )

        return train_dataloader, val_dataloader, test_dataloader

    def fit_models(self):
        train_dataloader, val_dataloader, test_dataloader = \
            self._create_dataloaders()

        for model in self.models:
            # train
            model.set_train_mode()
            train_loss = 0
            for x_batch, y_batch in train_dataloader:
                # print(x_batch.shape, y_batch.shape)
                loss = model._train_step(x_batch, y_batch)
                train_loss += loss

            print(f'Train loss: {train_loss / len(train_dataloader)}')

            model.set_test_mode()
            val_loss = 0
            for x_batch, y_batch in train_dataloader:
                loss = model._val_step(x_batch, y_batch)
                val_loss += loss

            print(f'Valid loss: {val_loss / len(val_dataloader)}')











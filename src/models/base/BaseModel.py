import os
import pathlib
from abc import abstractmethod, ABC
from uuid import uuid4

import pandas as pd
import torch

from models.logger.Logger import Logger


class BaseModel(ABC):
    def __init__(
            self,
            model,
            optimizer,
            lr,
            loss_fn,
            transforms,
            metrics,
            num_classes,
            num_epoch,
            device,
    ):
        self._torch_model = model
        self._optimizer = optimizer(self._torch_model.parameters(), lr=lr)
        self._loss_fn = loss_fn
        self._lr = lr
        self._num_epoch = num_epoch
        self._metrics = metrics
        self._transforms = transforms

        self._num_classes = num_classes
        self._curr_epoch = 0
        self._id = uuid4().hex[::5]
        self._device = device
        self._torch_model.to(self._device)

        self._tmp_path = pathlib.Path(os.getenv("TMP_PATH"))
        self._logger = Logger(self._tmp_path / 'logs' / str(self._id))

    def save(self):
        path = self._tmp_path / 'weights'
        if not os.path.exists(path):
            os.mkdir(path)
        torch.save(self._torch_model, path / f'{self._id}.pt')

    def _to_csv(self, hparams, scores):
        data = {"_id": self._id, "name": self.__str__()}
        data.update(hparams)
        for score in scores:
            data[score] = float(scores[score])

        path = self._tmp_path / 'models.csv'
        df = pd.read_csv(path) if path.exists() else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        df.to_csv(path, index=False)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @abstractmethod
    def train_step(self, train_ds):
        pass

    @abstractmethod
    def val_step(self, val_ds):
        pass

    @abstractmethod
    def predict(self, x_batch):
        pass

    def train_mode(self):
        self._torch_model.train()

    def eval_mode(self):
        self._torch_model.eval()

    @property
    def transforms(self):
        return self._transforms

    def __str__(self):
        return f'{self._id}_{self._torch_model.__class__.__name__}_{self._optimizer.__class__.__name__}_' \
               f'{self._loss_fn.__class__.__name__}_lr={str(self._lr).replace(".", ",")}'

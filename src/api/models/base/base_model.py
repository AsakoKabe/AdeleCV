from __future__ import annotations

import os
from abc import abstractmethod, ABC
from uuid import uuid4

import albumentations
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from api.logs import get_logger
from api.models.tensorboard_logger import TensorboardLogger
from config import get_settings


class BaseModel(ABC):
    def __init__(
            self,
            model: torch.nn.Module,
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
        self._device = device
        self._torch_model.to(self._device)
        self._optimizer = optimizer(self._torch_model.parameters(), lr=lr)
        self._loss_fn = loss_fn
        self._lr = lr
        self._num_epoch = num_epoch
        self._metrics = metrics
        self._transforms = transforms

        self._num_classes = num_classes
        self._curr_epoch = 0
        self._id = uuid4().hex[::5]

        self._weights_path = get_settings().WEIGHTS_PATH
        self._logger = TensorboardLogger(get_settings().TENSORBOARD_LOGS_PATH / str(self._id))
        self._stats_model = None

    def save_weights(self) -> None:
        path = self._weights_path
        if not os.path.exists(path):
            os.mkdir(path)
        save_path = path / f'{self._id}.pt'
        torch.save(self._torch_model, save_path)
        get_logger().debug("Save weights model: %s, path: %s", str(self), save_path.as_posix())

    @property
    def stats_model(self) -> pd.DataFrame:
        return self._stats_model

    def _save_stats_model(
            self,
            hparams: dict[str, str | float | int],
            scores: dict[str, float]
    ) -> None:
        self._stats_model = {"_id": self._id, "name": str(self)}
        self._stats_model.update(hparams)
        for score in scores:
            self._stats_model[score] = round(float(scores[score]), 3)

    def _get_hparams(self) -> dict[str, str | float | int]:
        return {
            'architecture': self._torch_model.__class__.__name__,
            'lr': self._lr,
            'optimizer': self._optimizer.__class__.__name__,
            'loss_fn': self._loss_fn.__class__.__name__,
            'num_epoch': self._num_epoch,
        }

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device):
        self._device = device

    @abstractmethod
    def train_step(self, train_ds: DataLoader):
        pass

    @abstractmethod
    def val_step(self, val_ds: DataLoader):
        pass

    @abstractmethod
    def predict(self, img: np.ndarray) -> None:
        pass

    def train_mode(self) -> None:
        self._torch_model.train()

    def eval_mode(self) -> None:
        self._torch_model.eval()

    @property
    def transforms(self) -> albumentations.Compose:
        return self._transforms

    def __str__(self) -> str:
        return f'{self._id}_{self._torch_model.__class__.__name__}_' \
               f'{self._optimizer.__class__.__name__}_' \
               f'{self._loss_fn.__class__.__name__}_lr={str(self._lr).replace(".", ",")}'

from abc import ABC, abstractmethod

import torch
from segmentation_models_pytorch.encoders import get_preprocessing_fn


from models.base.BaseModel import BaseModel
from models.semantic.utils import get_preprocessing


class SegmentationModel(BaseModel):
    def __init__(
            self,
            model,
            optimizer,
            lr: float,
            loss_fn,
            num_classes,
            device,
            encoder_name='resnet34',
            encoder_weights='imagenet',
    ):
        super().__init__()
        self.device = device
        self._torch_model = model(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
        )
        self._torch_model.to(self.device)
        self._transforms = get_preprocessing(
            get_preprocessing_fn(encoder_name, pretrained=encoder_weights)
        )
        self.optimizer = optimizer(self._torch_model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.lr = lr

    def train_step(self, x_batch, y_batch):
        self.optimizer.zero_grad()
        pred = self._torch_model(x_batch.to(self.device))
        loss = self.loss_fn(pred, y_batch.long().to(self.device))
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def val_step(self, x_batch, y_batch):
        with torch.no_grad():
            pred = self._torch_model(x_batch.to(self.device))
            loss = self.loss_fn(pred, y_batch.long().to(self.device))

        return loss.detach()

    def predict(self, img):
        img = self.transforms(image=img)['image']
        img = torch.Tensor(img).float().unsqueeze(0)

        with torch.no_grad():
            pred = self._torch_model(img.to(self.device))

        return pred

    def __str__(self):
        return f'{self.__class__.__name__}__{self.optimizer.__class__.__name__}__' \
               f'{self.loss_fn.__class__.__name__}__lr={str(self.lr).replace(".", ",")}'

    def train_mode(self):
        self._torch_model.train()

    def eval_mode(self):
        self._torch_model.eval()

    @property
    def transforms(self):
        return self._transforms

import torch
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead, \
    DeepLabV3_MobileNet_V3_Large_Weights

from .BaseSemanticModel import BaseSemanticModel


class DeepLabV3MobileNet(BaseSemanticModel):
    def __init__(
            self,
            optimizer,
            loss_fn,
            lr,
            num_classes,
            mode=''
    ):
        super().__init__(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr=lr,
            weights=DeepLabV3_MobileNet_V3_Large_Weights,
            model=models.segmentation.deeplabv3_mobilenet_v3_large,
        )
        self.model.classifier = DeepLabHead(960, num_classes)
        # self.model.aux_classifier = FCNHead(40, num_classes)

    def train_step(self, x_batch, y_batch):
        self.optimizer.zero_grad()
        pred = self.model(x_batch)['out']
        loss = self.loss_fn(pred, y_batch)
        loss.backward()
        self.optimizer.step()

        return loss.detach()

    def val_step(self, x_batch, y_batch):
        with torch.no_grad():
            pred = self.model(x_batch)['out']
            loss = self.loss_fn(pred, y_batch).detach()

        return loss

    def predict(self, x_batch):
        raise NotImplemented

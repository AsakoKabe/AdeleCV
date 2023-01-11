import torch
from torchvision import models
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.lraspp import LRASPPHead

from models.semantic.BaseSemanticModel import BaseSemanticModel


class LRASPPMobileNetV3(BaseSemanticModel):
    def __init__(
            self,
            optimizer,
            loss_fn,
            lr,
            num_classes=37,
            mode=''
    ):
        super().__init__(
            optimizer=optimizer,
            loss_fn=loss_fn,
            lr=lr,
            weights=LRASPP_MobileNet_V3_Large_Weights,
            model=models.segmentation.lraspp_mobilenet_v3_large
        )

        self.model.classifier = LRASPPHead(
            low_channels=40,
            high_channels=960,
            inter_channels=128,
            num_classes=num_classes
        )

    def _train_step(self, x_batch, y_batch):
        self.optimizer.zero_grad()
        pred = self.model(x_batch)['out']
        loss = self.loss_fn(y_batch, pred)
        loss.backward()
        self.optimizer.step()

        return loss.detach().numpy()

    def _val_step(self, x_batch, y_batch):
        with torch.no_grad():
            pred = self.model(x_batch)['out']
            loss = self.loss_fn(y_batch, pred).detach().numpy()

        return loss

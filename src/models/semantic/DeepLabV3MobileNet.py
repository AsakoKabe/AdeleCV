from torchvision import models
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead, \
    DeepLabV3_MobileNet_V3_Large_Weights

from models.semantic.BaseSemanticModel import BaseSemanticModel


class DeepLabV3MobileNet(BaseSemanticModel):
    def __init__(
            self,
            optimizer,
            loss_fn,
            num_classes=2,
            mode=''
    ):
        super(BaseSemanticModel, self).__init__(
            optimizer=optimizer,
            loss_fn=loss_fn,
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
            model=models.segmentation.deeplabv3_mobilenet_v3_large,
        )

        self.model.classifier = DeepLabHead(960, num_classes)
        self.model.aux_classifier = FCNHead(40, num_classes)

from torchvision import models
from torchvision.models.segmentation import LRASPP_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.lraspp import LRASPPHead

from models.semantic.BaseSemanticModel import BaseSemanticModel


class DeepLabV3MobileNet(BaseSemanticModel):
    def __init__(
            self,
            num_classes=2,
            mode=''
    ):
        super().__init__(
            weights=LRASPP_MobileNet_V3_Large_Weights.DEFAULT,
            model=models.segmentation.lraspp_mobilenet_v3_large
        )

        self.model.classifier = LRASPPHead(
            low_channels=40,
            high_channels=128,
            inter_channels=128,
            num_classes=num_classes)

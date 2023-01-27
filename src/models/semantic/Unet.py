# import torch
# from torchvision import models
# from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead, \
#     DeepLabV3_MobileNet_V3_Large_Weights
#
# from .SemanticModel import SemanticModel
# from .UNet_pytorch import UNetTorch
#
#
# class Unet(SemanticModel):
#     def __init__(
#             self,
#             optimizer,
#             loss_fn,
#             lr,
#             num_classes,
#             mode=''
#     ):
#         super().__init__(
#             optimizer=optimizer,
#             loss_fn=loss_fn,
#             lr=lr,
#             model=UNetTorch(3, num_classes),
#         )
#
#     def train_step(self, x_batch, y_batch):
#         self.optimizer.zero_grad()
#         pred = self.model(x_batch)['out']
#         loss = self.loss_fn(pred, y_batch)
#         loss.backward()
#         self.optimizer.step()
#
#         return loss.detach()
#
#     def val_step(self, x_batch, y_batch):
#         with torch.no_grad():
#             pred = self.model(x_batch)['out']
#             loss = self.loss_fn(pred, y_batch).detach()
#
#         return loss
#
#     def predict(self, x_batch):
#         with torch.no_grad():
#             pred = self.model(x_batch)['out']
#
#         return pred

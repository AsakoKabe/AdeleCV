import torch


def bce_loss(y_real, y_pred):
    return torch.mean(
        y_pred - y_pred * y_real + torch.log(1 + torch.exp(-y_pred)))


def dice_loss(y_real, y_pred):
    smooth = 1e-8
    y_pred = torch.sigmoid(y_pred)

    num = torch.sum(y_pred * y_real)
    den = torch.sum(y_pred + y_real)
    res = torch.mean(1 - (2 * num / (den + smooth)))

    return res

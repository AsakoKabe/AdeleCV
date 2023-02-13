from segmentation_models_pytorch import Unet, UnetPlusPlus, MAnet, Linknet, FPN, PSPNet, DeepLabV3, DeepLabV3Plus, PAN
from segmentation_models_pytorch.losses import JaccardLoss, DiceLoss, FocalLoss, LovaszLoss, SoftBCEWithLogitsLoss, \
    SoftCrossEntropyLoss, TverskyLoss, MCCLoss
from segmentation_models_pytorch.metrics import fbeta_score, f1_score, iou_score, accuracy, precision, recall
import segmentation_models_pytorch as smp
from torch.optim import AdamW, Adadelta, Adam, SGD, RAdam, NAdam, RMSprop, Adagrad

from .SegmentationModel import SegmentationModel

__all__ = [
    "SegmentationModel",
    "get_segmentations_models",
    "get_encoders",
    "get_pretrained_weights",
    "get_torch_optimizers",
    'get_optimize_scores',
    'get_losses'
]

_segmentations_models = [
    Unet,
    UnetPlusPlus,
    MAnet,
    Linknet,
    FPN,
    PSPNet,
    DeepLabV3,
    DeepLabV3Plus,
    PAN,
]

_pretrained_weights = ['imagenet', 'None']

_torch_optimizers = [
    AdamW,
    Adadelta,
    Adam,
    SGD,
    RAdam,
    NAdam,
    RMSprop,
    Adagrad,
]

_losses = [
    JaccardLoss,
    DiceLoss,
    FocalLoss,
    LovaszLoss,
    SoftBCEWithLogitsLoss,
    SoftCrossEntropyLoss,
    TverskyLoss,
    MCCLoss,
]

_optimize_scores = [
    fbeta_score,
    f1_score,
    iou_score,
    accuracy,
    precision,
    recall,
]


def get_obj_names(classes: list):
    return list(map(lambda cls: cls.__name__, classes))


def get_segmentations_models():
    return get_obj_names(_segmentations_models)


def get_encoders():
    return smp.encoders.get_encoder_names()


def get_pretrained_weights():
    return _pretrained_weights


def get_torch_optimizers():
    return get_obj_names(_torch_optimizers)


def get_losses():
    return get_obj_names(_losses)


def get_optimize_scores():
    scores = get_obj_names(_optimize_scores)
    scores.append('loss')

    return scores

import segmentation_models_pytorch as smp
from segmentation_models_pytorch import (FPN, PAN, DeepLabV3, DeepLabV3Plus,
                                         Linknet, MAnet, PSPNet, Unet,
                                         UnetPlusPlus)
from segmentation_models_pytorch.losses import (DiceLoss, FocalLoss,
                                                JaccardLoss, LovaszLoss,
                                                MCCLoss, SoftBCEWithLogitsLoss,
                                                SoftCrossEntropyLoss,
                                                TverskyLoss)
from segmentation_models_pytorch.metrics import (accuracy, f1_score,
                                                 fbeta_score, iou_score,
                                                 precision, recall)
from torch.optim import (SGD, Adadelta, Adagrad, Adam, AdamW, NAdam, RAdam,
                         RMSprop)

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


def get_obj_names(objs: list) -> list[str]:
    return [obj.__name__ for obj in objs]


def get_models() -> list[str]:
    """
    :return: List of names model
    """
    return get_obj_names(_segmentations_models)


def get_encoders() -> list[str]:
    """

    :return: List of names encoders
    """
    return smp.encoders.get_encoder_names()


def get_pretrained_weights() -> list[str]:
    """

    :return: List of names pretrained weights
    """
    return _pretrained_weights


def get_torch_optimizers() -> list[str]:
    """

    :return: List of names torch optimizers
    """
    return get_obj_names(_torch_optimizers)


def get_losses() -> list[str]:
    """

    :return: List of names losses
    """
    return get_obj_names(_losses)


def get_optimize_scores() -> list[str]:
    """

    :return: List of names optimize scores
    """
    scores = get_obj_names(_optimize_scores)
    scores.append('loss')

    return scores

from __future__ import annotations

from pydantic import BaseModel, validator

from adelecv.api.models.segmentations import (get_encoders, get_losses,
                                              get_models, get_optimize_scores,
                                              get_pretrained_weights,
                                              get_torch_optimizers)

from .functional import get_hp_optimizers


def _check_contains_and_empty(
        values: list[str], possible_values: list[str], param_name: str
        ):
    if not values:
        raise ValueError(f"{param_name} can't be None")
    for value in values:
        if value not in possible_values:
            raise ValueError(
                f"{param_name} can't contain {value}, only: {possible_values}"
                )

    return values


def _check_range(value: tuple[float | int, float | int], param_name: str):
    if len(value) != 2:
        raise ValueError(
            f"{param_name} must contain 2 value (left and right range)"
            )
    if value[0] > value[1]:
        raise ValueError(f"right range {param_name} can't be more left range")

    return value


def _check_contain_and_none(
        value: str, possible_values: list[str], param_name: str, ):
    if not value:
        raise ValueError(f"{param_name} can't be None")
    if value not in possible_values:
        raise ValueError(
            f"{param_name} can't contain {value}, only: {possible_values}"
            )
    return value


class HyperParamsSegmentation(BaseModel):
    """
    Dataclass with set of hyperparams. For all possible
    values see :ref:`model-segmentations`.

    :param strategy: Name optimizer strategy (optuna name Sampler).
    :param architectures: List of name model (Unet, DeepLabV3, e.g.)
    :param encoders: List of name encoders (resnet18, mobilenet, e.g.)
    :param pretrained_weights: List of pretrained weights (imagenet or None),
     only str! None='None'
    :param loss_fns: List of names loss func (DiceLoss, JaccardLoss, e.g.)
    :param optimizers: List of names pytorch optimizers (AdamW, Adadelta, e.g.)
    :param epoch_range: range epoch (from - to)
    :param lr_range: range lr (from - to)
    :param optimize_score: Score for optimizing optimizers (optuna score)
    """
    strategy: str
    architectures: list[str]
    encoders: list[str]
    pretrained_weights: list[str]
    loss_fns: list[str]
    optimizers: list[str]
    epoch_range: tuple[int, int]
    lr_range: tuple[float, float]
    optimize_score: str

    @validator('strategy')
    @classmethod
    def check_strategy(cls, v):
        return _check_contain_and_none(v, get_hp_optimizers(), 'strategy')

    @validator('architectures')
    @classmethod
    def check_architectures(cls, v):
        return _check_contains_and_empty(v, get_models(), 'architectures')

    @validator('encoders')
    @classmethod
    def check_encoders(cls, v):
        return _check_contains_and_empty(v, get_encoders(), 'encoders')

    @validator('pretrained_weights')
    @classmethod
    def check_pretrained_weights(cls, v):
        return _check_contains_and_empty(
            v, get_pretrained_weights(), 'pretrained_weights'
            )

    @validator('loss_fns')
    @classmethod
    def check_loss_fns(cls, v):
        return _check_contains_and_empty(v, get_losses(), 'loss_fns')

    @validator('optimizers')
    @classmethod
    def check_optimizers(cls, v):
        return _check_contains_and_empty(
            v, get_torch_optimizers(), 'optimizers'
            )

    @validator('epoch_range')
    @classmethod
    def check_epoch_range(cls, v):
        return _check_range(v, 'epoch_range')

    @validator('lr_range')
    @classmethod
    def check_lr_range(cls, v):
        return _check_range(v, 'lr_range')

    @validator('optimize_score')
    @classmethod
    def check_optimize_score(cls, v):
        return _check_contain_and_none(
            v, get_optimize_scores(), 'optimize_score'
            )

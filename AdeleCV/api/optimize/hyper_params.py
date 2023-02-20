from __future__ import annotations

from pydantic import BaseModel, validator

from api.models.segmentations import get_models, get_encoders, get_pretrained_weights, \
    get_torch_optimizers, get_losses, get_optimize_scores
from .utils import get_hp_optimizers


def _check_contains_and_empty(values: list[str], possible_values: list[str], param_name: str):
    if not values:
        raise ValueError(f"{param_name} can't be None")
    for value in values:
        if value not in possible_values:
            raise ValueError(f"{param_name} can't contain {value}, only: {possible_values}")

    return values


def _check_range(value: tuple[float | int, float | int], param_name: str):
    if len(value) != 2:
        raise ValueError(f"{param_name} must contain 2 value (left and right range)")
    if value[0] > value[1]:
        raise ValueError(f"right range {param_name} can't be more left range")

    return value


def _check_contain_and_none(value: str, possible_values: list[str], param_name: str, ):
    if not value:
        raise ValueError(f"{param_name} can't be None")
    if value not in possible_values:
        raise ValueError(f"{param_name} can't contain {value}, only: {possible_values}")
    return value


class HyperParamsSegmentation(BaseModel):
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
        return _check_contains_and_empty(v, get_pretrained_weights(), 'pretrained_weights')

    @validator('loss_fns')
    @classmethod
    def check_loss_fns(cls, v):
        return _check_contains_and_empty(v, get_losses(), 'loss_fns')

    @validator('optimizers')
    @classmethod
    def check_optimizers(cls, v):
        return _check_contains_and_empty(v, get_torch_optimizers(), 'optimizers')

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
        return _check_contain_and_none(v, get_optimize_scores(), 'optimize_score')

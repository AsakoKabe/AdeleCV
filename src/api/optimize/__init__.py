from optuna.samplers import RandomSampler, GridSampler, TPESampler, CmaEsSampler, NSGAIISampler, QMCSampler, \
    MOTPESampler

from .hp_optimizer import HPOptimizer


__all__ = [
    "HPOptimizer",
    "get_hp_optimizers",
]

_hp_optimizers = [
    RandomSampler,
    GridSampler,
    TPESampler,
    CmaEsSampler,
    NSGAIISampler,
    QMCSampler,
    MOTPESampler,
]


def get_hp_optimizers():
    return list(map(lambda cls: cls.__name__, _hp_optimizers))

from optuna.samplers import (CmaEsSampler, GridSampler, MOTPESampler,
                             NSGAIISampler, QMCSampler, RandomSampler,
                             TPESampler)

_hp_optimizers = [
    RandomSampler,
    GridSampler,
    TPESampler,
    CmaEsSampler,
    NSGAIISampler,
    QMCSampler,
    MOTPESampler,
]


def get_hp_optimizers() -> list[str]:
    """
    Get list of names optimizer strategy

    :return: List of names hyperparams optimizer strategy
    """
    return [obj.__name__ for obj in _hp_optimizers]

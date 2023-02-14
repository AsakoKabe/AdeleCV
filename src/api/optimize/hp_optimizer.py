import pandas as pd
import torch
from torch import optim
import optuna
from optuna.trial import TrialState

from optuna import samplers, Study, Trial
import segmentation_models_pytorch as smp

from api.data.segmentations import SegmentationDataset
from api.logs import get_logger
from api.models.segmentations import SegmentationModel


def _log_study(study: Study) -> None:
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    logger = get_logger()
    logger.info("Study statistics:")
    logger.info("Number of finished trials: %s", len(study.trials))
    logger.info("Number of pruned trials: %s", len(pruned_trials))
    logger.info("Number of complete trials: %s", len(complete_trials))

    logger.info("Best trial:")
    trial = study.best_trial

    logger.info("Value: %s", trial.value)

    logger.info("Params: ")
    for key, value in trial.params.items():
        logger.info("%s: %s", key, value)


class HPOptimizer:
    def __init__(
            self,
            architectures: list[str],
            encoders: list[str],
            pretrained_weights: list[str],
            lr_range: tuple[float, float],
            optimizers: list[str],
            loss_fns: list[str],
            epoch_range: tuple[int, int],
            strategy: str,
            num_trials: int,
            device: str,
            dataset: SegmentationDataset,
            optimize_score: str,
    ):
        # todo: optimizer для разных тасок
        self.lr_range = lr_range
        self.optimizers = optimizers
        self.architectures = architectures
        self.encoders = encoders
        self._pretrained_weights = pretrained_weights
        self.epoch_range = epoch_range
        self.loss_fns = loss_fns
        self.strategy = strategy
        self.num_trials = num_trials
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device == 'GPU'
            else 'cpu'
        )
        self._stats_models = pd.DataFrame()
        self._optimize_score = optimize_score

    def optimize(self) -> None:
        direction = 'minimize' if self._optimize_score == 'loss' else 'maximize'
        study = optuna.create_study(
            direction=direction,
            sampler=getattr(samplers, self.strategy)()
        )
        study.optimize(self._objective, n_trials=self.num_trials, timeout=600)
        _log_study(study)

    def _create_model(self, trial: Trial) -> tuple[SegmentationModel, int]:
        optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)
        optimizer = getattr(optim, optimizer_name)
        architecture_name = trial.suggest_categorical("architecture", self.architectures)
        architecture = getattr(smp, architecture_name)
        encoder = trial.suggest_categorical("encoders", self.encoders)
        pretrained_weight = trial.suggest_categorical("pretrained_weight", self._pretrained_weights)
        lr = trial.suggest_float("lr", self.lr_range[0], self.lr_range[1])
        loss_name = trial.suggest_categorical("loss", self.loss_fns)
        loss_fn = getattr(smp.losses, loss_name)('binary')
        num_epoch = trial.suggest_int("num_epoch", self.epoch_range[0], self.epoch_range[1])

        model = SegmentationModel(
            model=architecture,
            encoder_name=encoder,
            pretrained_weight=pretrained_weight if pretrained_weight != "None" else None,
            optimizer=optimizer,
            lr=lr,
            loss_fn=loss_fn,
            num_classes=self.num_classes,
            num_epoch=num_epoch,
            device=self.device,
            img_size=self.dataset.img_size,
        )

        return model, num_epoch

    def _objective(self, trial: Trial) -> float:
        model, num_epoch = self._create_model(trial)
        self.dataset.update_datasets(model.transforms)

        score = 0
        for epoch in range(num_epoch):
            score = self._train_model(model)
            trial.report(score, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self._postprocessing_model(model)

        return score

    def _train_model(self, model: SegmentationModel) -> float:
        torch.cuda.empty_cache()
        model.train_step(self.dataset.train)
        val_loss = model.val_step(self.dataset.val)

        return val_loss[self._optimize_score]

    def _postprocessing_model(self, model: SegmentationModel) -> None:
        model.log_test(self.dataset.test)
        self.dataset.add_predictions(model)
        self._add_stats_model(model)
        model.save_weights()

    def _add_stats_model(self, model: SegmentationModel) -> None:
        self._stats_models = pd.concat(
            [self._stats_models, pd.DataFrame([model.stats_model])],
            ignore_index=True
        )

    @property
    def stats_models(self) -> pd.DataFrame:
        return self._stats_models

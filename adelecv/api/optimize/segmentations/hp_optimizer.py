import optuna
import pandas as pd
import segmentation_models_pytorch as smp
import torch
from optuna import Study, Trial, samplers
from optuna.trial import TrialState
from torch import optim

from adelecv.api.data.segmentations import SegmentationDataset
from adelecv.api.logs import get_logger
from adelecv.api.models.segmentations import SegmentationModel

from .hyper_params import HyperParamsSegmentation


def _log_study(study: Study) -> None:
    pruned_trials = study.get_trials(
        deepcopy=False, states=[TrialState.PRUNED]
    )
    complete_trials = study.get_trials(
        deepcopy=False, states=[TrialState.COMPLETE]
    )

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
    """
    Class for hyperparams search and model training

    :param hyper_params: Dataclass with hyperparams of models
    :param num_trials: Number of iterations algorithm (the number of models
     with pruned models)
    :param device: GPU or CPU
    :param dataset: Created dataset
    """

    def __init__(
            self,
            hyper_params: HyperParamsSegmentation,
            num_trials: int,
            device: str,
            dataset: SegmentationDataset,
    ):
        self._hyper_params = hyper_params
        self._strategy = self._hyper_params.strategy
        self._num_trials = num_trials
        self._dataset = dataset
        self._num_classes = dataset.num_classes
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() and device == 'GPU'
            else 'cpu'
        )
        get_logger().debug(
            "Create hp optimizer with params: "
            "strategy: %s num_trials: %s num_classes: %s "
            "device: %s hps: %s",
            self._strategy, self._num_trials, self._num_classes,
            self._device, self._hyper_params.dict()
        )
        self._stats_models = pd.DataFrame()

    def optimize(self) -> None:
        """
        Run optimize.

        """
        get_logger().info("Train models started")

        direction = 'minimize' if self._hyper_params.optimize_score == 'loss' \
            else 'maximize'
        study = optuna.create_study(
            direction=direction,
            sampler=getattr(samplers, self._strategy)()
        )
        study.optimize(self._objective, n_trials=self._num_trials, timeout=600)
        _log_study(study)

        get_logger().info("Train models is over")

    def _create_model(self, trial: Trial) -> tuple[SegmentationModel, int]:
        optimizer_name = trial.suggest_categorical(
            "optimizer", self._hyper_params.optimizers
        )
        optimizer = getattr(optim, optimizer_name)
        architecture_name = trial.suggest_categorical(
            "architecture", self._hyper_params.architectures
        )
        architecture = getattr(smp, architecture_name)
        encoder = trial.suggest_categorical(
            "encoders", self._hyper_params.encoders
        )
        pretrained_weight = trial.suggest_categorical(
            "pretrained_weight", self._hyper_params.pretrained_weights
        )
        lr = trial.suggest_float(
            "lr", self._hyper_params.lr_range[0],
            self._hyper_params.lr_range[1]
        )
        loss_name = trial.suggest_categorical(
            "loss", self._hyper_params.loss_fns
        )
        loss_fn = getattr(smp.losses, loss_name)('binary')
        num_epoch = trial.suggest_int(
            "num_epoch", self._hyper_params.epoch_range[0],
            self._hyper_params.epoch_range[1]
        )

        model = SegmentationModel(
            model=architecture,
            encoder_name=encoder,
            pretrained_weight=pretrained_weight if pretrained_weight != "None"
            else None,
            optimizer=optimizer,
            lr=lr,
            loss_fn=loss_fn,
            num_classes=self._num_classes,
            num_epoch=num_epoch,
            device=self._device,
            img_size=self._dataset.img_size,
        )

        return model, num_epoch

    def _objective(self, trial: Trial) -> float:
        model, num_epoch = self._create_model(trial)
        self._dataset.update_datasets(model.transforms)

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
        model.train_step(self._dataset.train)
        val_loss = model.val_step(self._dataset.val)

        return val_loss[self._hyper_params.optimize_score]

    def _postprocessing_model(self, model: SegmentationModel) -> None:
        model.log_test(self._dataset.test)
        self._dataset.add_predictions(model)
        self._add_stats_model(model)
        model.save_weights()

    def _add_stats_model(self, model: SegmentationModel) -> None:
        self._stats_models = pd.concat(
            [self._stats_models, pd.DataFrame([model.stats_model])],
            ignore_index=True
        )

    @property
    def stats_models(self) -> pd.DataFrame:
        """
        Get stats models after training.

        :return: Info about trained models
        """
        return self._stats_models

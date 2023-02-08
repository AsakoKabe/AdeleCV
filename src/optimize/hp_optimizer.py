import os

import pandas as pd
import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState

from optuna import samplers
import segmentation_models_pytorch as smp

from models.semantic.SegmentationModel import SegmentationModel


class HPOptimizer:
    def __init__(
            self,
            architectures,
            encoders,
            lr_range,
            optimizers,
            loss_fns,
            epoch_range,
            strategy,
            num_trials,
            device,
            dataset,
    ):
        # todo: optimizer для разных тасок
        self.lr_range = lr_range
        self.optimizers = optimizers
        self.architectures = architectures
        self.encoders = encoders
        self.epoch_range = epoch_range
        self.loss_fns = loss_fns
        self.strategy = strategy
        self.num_trials = num_trials
        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')
        self._stats_models = pd.DataFrame()

    def optimize(self):
        study = optuna.create_study(
            direction="minimize",
            sampler=getattr(samplers, self.strategy)()
        )
        study.optimize(self._objective, n_trials=self.num_trials, timeout=600)

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

    def _create_model(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)
        optimizer = getattr(optim, optimizer_name)
        architecture_name = trial.suggest_categorical("architecture", self.architectures)
        architecture = getattr(smp, architecture_name)
        encoder = trial.suggest_categorical("encoders", self.encoders)
        # encoder = getattr(smp, architecture_name)
        lr = trial.suggest_float("lr", self.lr_range[0], self.lr_range[1])
        loss_name = trial.suggest_categorical("loss", self.loss_fns)
        loss_fn = getattr(smp.losses, loss_name)('binary')
        num_epoch = trial.suggest_int("num_epoch", self.epoch_range[0], self.epoch_range[1])

        model = SegmentationModel(
            model=architecture,
            encoder_name=encoder,
            optimizer=optimizer,
            lr=lr,
            loss_fn=loss_fn,
            num_classes=self.num_classes,
            num_epoch=num_epoch,
            device=self.device,
            img_size=self.dataset.img_size,
        )

        return model, num_epoch

    def _objective(self, trial):
        model, num_epoch = self._create_model(trial)
        self.dataset.update_datasets(model.transforms)

        loss = 0
        for epoch in range(num_epoch):
            loss = self._train_model(model)
            trial.report(loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        self._postprocessing_model(model)

        return loss

    def _train_model(self, model: SegmentationModel):
        torch.cuda.empty_cache()
        model.train_step(self.dataset.train)
        val_loss = model.val_step(self.dataset.val)

        return val_loss

    def _postprocessing_model(self, model):
        model.log_test(self.dataset.test)
        self.dataset.add_predictions(model)
        self._add_stats_model(model)
        model.save_weights()

    def _add_stats_model(self, model: SegmentationModel):
        self._stats_models = pd.concat(
            [self._stats_models, pd.DataFrame([model.stats_model])],
            ignore_index=True
        )

    @property
    def stats_models(self):
        return self._stats_models

import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch import nn
from torchmetrics.functional.classification import multiclass_jaccard_index

import models.semantic as semantic_models
import optuna.samplers as samplers


class HPOptimizer:
    def __init__(
            self,
            architectures,
            lr_range,
            optimizers,
            loss_fns,
            epoch_range,
            strategy,
            num_trials,
            device,
            task,
    ):
        self.lr_range = lr_range
        self.optimizers = optimizers
        self.architectures = architectures
        self.epoch_range = epoch_range
        self.loss_fns = loss_fns
        self.strategy = strategy
        self.num_trials = num_trials
        self.task = task
        self.num_classes = task.dataset.num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() and device == 'GPU' else 'cpu')

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

    def _objective(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)
        optimizer = getattr(optim, optimizer_name)
        architecture_name = trial.suggest_categorical("architecture", self.architectures)
        architecture = getattr(semantic_models, architecture_name)
        lr = trial.suggest_float("lr", self.lr_range[0], self.lr_range[1])
        loss_name = trial.suggest_categorical("loss", self.loss_fns)
        loss_fn = getattr(nn, loss_name)()
        num_epoch = trial.suggest_int("num_epoch", self.epoch_range[0], self.epoch_range[1])

        model = architecture(
            optimizer,
            loss_fn,
            lr,
            self.num_classes
        )

        loss = 0
        for epoch in range(num_epoch):
            loss = self._train_model(model)
            trial.report(loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        test_iou = 0
        for x_batch, y_batch in self.task.dataset.test:
            pred = model.predict(x_batch.to(self.device))
            pred = (nn.Sigmoid()(pred) >= 0.5).float()
            # y_batch = torch.argmax(y_batch, dim=1)
            test_iou += multiclass_jaccard_index(pred, y_batch.to(self.device), num_classes=2)
        print(test_iou)

        self.task.dataset.add_predictions(model)

        return loss

    def _train_model(self, model):
        model.set_device(self.device)
        model.set_train_mode()
        train_loss = 0
        for x_batch, y_batch in self.task.dataset.train:
            loss = model.train_step(x_batch.to(self.device), y_batch.to(self.device))
            train_loss += loss.cpu().numpy()

        model.set_test_mode()
        val_loss = 0
        for x_batch, y_batch in self.task.dataset.val:
            loss = model.val_step(x_batch.to(self.device), y_batch.to(self.device))
            val_loss += loss.cpu().numpy()

        return val_loss / len(self.task.dataset.val)

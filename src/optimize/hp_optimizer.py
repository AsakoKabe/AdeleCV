import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch import nn
import models.semantic as semantic_models


class HPOptimizer:
    def __init__(self, task):
        self.lrs = [0.003, 0.01, 0.001]
        self.optimizers = ["Adam", "RMSprop", "SGD"]
        self.architectures = ["DeepLabV3MobileNet", "LRASPPMobileNetV3"]
        self.num_epoch = 2
        self.task = task
        self.num_classes = task.dataset.num_classes
        self.device = task.device

    def optimize(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self._objective, n_trials=10, timeout=600)

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
        lr = trial.suggest_categorical("lr", self.lrs)

        model = architecture(
            optimizer,
            nn.CrossEntropyLoss(),
            lr,
            self.num_classes
        )

        loss = 0
        for epoch in range(self.num_epoch):
            loss = self._train_model(model)
            trial.report(loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

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

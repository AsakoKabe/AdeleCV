import torch
import torch.optim as optim
import optuna
from optuna.trial import TrialState
from torch import nn
from torchmetrics.functional.classification import multiclass_jaccard_index

from models.semantic.DeepLabV3MobileNet import DeepLabV3MobileNet


class HPOptimizer:
    def __init__(self, task):
        self.lrs = [0.003, 0.01, 0.001]
        self.optimizers = ["Adam", "RMSprop", "SGD"]
        self.num_epoch = 2
        self.task = task
        self.num_classes = task.dataset.num_classes
        self.device = task.device

    def optimize(self):
        study = optuna.create_study(direction="minimize")
        study.optimize(self.__objective, n_trials=10, timeout=600)

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

    def __objective(self, trial):
        optimizer_name = trial.suggest_categorical("optimizer", self.optimizers)
        lr = trial.suggest_categorical("lr", self.lrs)
        optimizer = getattr(optim, optimizer_name)
        # todo: понять как итерироваться по моделям
        model = DeepLabV3MobileNet(
            optimizer,
            nn.CrossEntropyLoss(),
            lr,
            self.num_classes
        )

        loss = 0
        for epoch in range(self.num_epoch):
            loss = self.__train_model(model)
            trial.report(loss, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return loss

    def __train_model(self, model):
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

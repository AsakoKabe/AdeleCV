from abc import ABC, abstractmethod

from models.base.BaseModel import BaseModel


class SemanticModel(BaseModel, ABC):
    def __init__(
            self,
            weights,
            model,
            optimizer,
            lr: float,
            loss_fn,
            scheduler=None
    ):
        super().__init__()
        self.weights = weights.DEFAULT
        self.model = model(
            weights=self.weights,
        )
        self.transforms = self.weights.transforms()
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.lr = lr

    def __str__(self):
        return f'{self.__class__.__name__}__{self.optimizer.__class__.__name__}__' \
               f'{self.loss_fn.__class__.__name__}__lr={str(self.lr).replace(".", ",")}'

    def set_train_mode(self):
        self.model.train()

    def set_test_mode(self):
        self.model.eval()

    def set_device(self, device):
        self.model.to(device)
        self.device = device

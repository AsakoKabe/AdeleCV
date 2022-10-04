from abc import ABC

from models.base.BaseModel import BaseModel


class BaseSemanticModel(BaseModel, ABC):
    def __init__(
            self,
            weights,
            model,
            optimizer,
            lr,
            loss_fn,
            scheduler=None
    ):
        self.weights = weights.DEFAULT
        self.model = model(
            weights=self.weights
        ).cuda()
        self.transforms = self.weights.transforms()
        self.optimizer = optimizer(self.model.parameters(), lr=lr)
        self.loss_fn = loss_fn
        self.lr = lr

    def __str__(self):
        return f'{self.__class__.__name__}__{self.optimizer.__name__}__' \
               f'{self.loss_fn.__name__}__lr={self.lr}'

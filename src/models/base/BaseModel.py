from abc import abstractmethod, ABC
from uuid import uuid4

from models.logger.Logger import Logger


class BaseModel(ABC):
    def __init__(
            self,
            model,
            optimizer,
            lr,
            loss_fn,
            transforms,
            metrics,
            num_classes,
            num_epoch,
            device,
    ):
        self._torch_model = model
        self._optimizer = optimizer(self._torch_model.parameters(), lr=lr)
        self._loss_fn = loss_fn
        self._lr = lr
        self._num_epoch = num_epoch
        self._metrics = metrics
        self._transforms = transforms

        self._num_classes = num_classes
        self._curr_epoch = 0
        self._id = uuid4().hex[::5]
        self._logger = Logger(f'../logs/{self._id}')
        self._device = device
        self._torch_model.to(self._device)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @abstractmethod
    def train_step(self, train_ds):
        pass

    @abstractmethod
    def val_step(self, val_ds):
        pass

    @abstractmethod
    def predict(self, x_batch):
        pass

    def train_mode(self):
        self._torch_model.train()

    def eval_mode(self):
        self._torch_model.eval()

    @property
    def transforms(self):
        return self._transforms

    def __str__(self):
        return f'{self._id}_{self._torch_model.__class__.__name__}_{self._optimizer.__class__.__name__}_' \
               f'{self._loss_fn.__class__.__name__}_lr={str(self._lr).replace(".", ",")}'

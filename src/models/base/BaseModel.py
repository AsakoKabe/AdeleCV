from abc import abstractmethod, ABC


class BaseModel(ABC):
    def __init__(self):
        self._device = None

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, device):
        self._device = device

    @abstractmethod
    def train_step(self, x_batch, y_batch):
        pass

    @abstractmethod
    def val_step(self, x_batch, y_batch):
        pass

    @abstractmethod
    def predict(self, x_batch):
        pass

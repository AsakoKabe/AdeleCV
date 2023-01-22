from abc import abstractmethod, ABC


class BaseTask(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
        self._device = None

    @abstractmethod
    def fit_models(self):
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, new_device):
        self._device = new_device

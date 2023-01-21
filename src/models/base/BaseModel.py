from abc import abstractmethod, ABC


class BaseModel(ABC):

    @abstractmethod
    def train_step(self, x_batch, y_batch):
        pass

    @abstractmethod
    def val_step(self, x_batch, y_batch):
        pass

    @abstractmethod
    def predict(self, x_batch):
        pass

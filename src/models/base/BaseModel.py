from abc import abstractmethod, ABC


class BaseModel(ABC):

    @abstractmethod
    def _train_step(self, x_batch, y_batch):
        pass

    @abstractmethod
    def _val_step(self, x_batch, y_batch):
        pass

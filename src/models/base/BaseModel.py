from abc import abstractmethod, ABC


class BaseModel(ABC):

    @abstractmethod
    def __train_step(self, x_batch, y_batch):
        pass

    @abstractmethod
    def __val_step(self, x_batch, y_batch):
        pass

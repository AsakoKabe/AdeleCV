from abc import abstractmethod


class BaseModel:

    def __train_step(self, x_batch, y_batch):
        pass

    def __val_step(self, x_batch, y_batch):
        pass
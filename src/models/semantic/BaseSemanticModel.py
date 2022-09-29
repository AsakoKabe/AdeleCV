from models.base.BaseModel import BaseModel


class BaseSemanticModel(BaseModel):
    def __init__(
            self,
            weights,
            model,
    ):
        self.weights = weights.DEFAULT
        self.model = model(
            weights=self.weights
        )
        self.transforms = self.weights.transforms()

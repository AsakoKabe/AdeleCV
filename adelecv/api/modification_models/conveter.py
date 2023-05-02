import abc
from pathlib import Path

import torch


class BaseConverter(abc.ABC):
    def __init__(self, input_shape: list[int]):
        self._dummy_input = torch.zeros(input_shape)    # BxCxHxW

    @property
    def dummy_input(self):
        return self._dummy_input

    @abc.abstractmethod
    def convert(
            self,
            torch_model: torch.nn.Module,
            path_to_save_weights: Path
    ) -> None:
        pass


class TorchToOnnx(BaseConverter):
    def convert(
            self,
            torch_model: torch.nn.Module,
            path_to_save_weights
    ) -> None:
        torch.onnx.export(
            torch_model,  # model being run
            self.dummy_input,  # model input (or a tuple for multiple inputs)
            path_to_save_weights,
            # where to save the model (can be a file or file-like object)
            export_params=True,
            # store the trained parameter weights inside the model file
            opset_version=10,  # the ONNX version to export the model to
            do_constant_folding=True,
            # whether to execute constant folding for optimization
            input_names=['input'],  # the model's input names
            output_names=['output'],  # the model's output names
            dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                          'output': {0: 'batch_size'}}
        )

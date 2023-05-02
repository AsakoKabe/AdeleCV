from __future__ import annotations

import os
import zipfile
from pathlib import Path
from uuid import uuid4

import torch

from adelecv.api.config import Settings
from adelecv.api.logs import get_logger

from .conveter import TorchToOnnx


def _create_zip(
        converted_weights_path: Path,
        zip_path: Path
) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in converted_weights_path.rglob("*"):
            zip_file.write(
                entry, entry.relative_to(
                    converted_weights_path
                )
            )
    get_logger().info(
        "Create zip with converted weights, path: %s", zip_path.as_posix()
    )


class ConvertWeights:
    """
    Class for conversation weights.

    :param weights_path: Path to saved weights.
    """

    def __init__(
            self,
            img_shape: list[int] | tuple[int],          # HxWxC
            weights_path: Path = Settings.WEIGHTS_PATH,

    ):
        if len(img_shape) != 3:
            raise ValueError("Input shape must be in the format HxWxC")

        # BxCxHxW
        self._input_shape = (1, img_shape[2], img_shape[0], img_shape[1])
        self._weights_path = weights_path
        self._supported_formats = ['onnx']
        self._converter = {
            'onnx': TorchToOnnx(self._input_shape)
        }

    def run(
            self,
            id_selected: None | set[str] | list[str] = None,
            new_format: None | str = None
    ) -> Path:
        """
        Converting selected models to the specified format.

        :param new_format: format weights for conversation
        :param id_selected: List with id models from stats_models
        :return: Path to created zip file with other formats weights.
        """

        if new_format not in self.supported_formats:
            raise ValueError(
                f"{new_format} format is not supported for conversion. "
                f"Supported formats: {self.supported_formats}"
            )

        id_convert = uuid4().hex
        path_to_save = self.weights_path.parent / f'converted_{id_convert}'
        os.mkdir(path_to_save.as_posix())
        path_to_zip = self.weights_path.parent / f'converted_{id_convert}.zip'

        for id_model in id_selected:
            get_logger().info(
                "Сonvert weights model: %s to %s format",
                id_model, new_format
            )
            path_weights = self.weights_path / f'{id_model}.pt'
            torch_model = torch.load(path_weights)
            torch_model.eval()
            self._convert(torch_model, new_format, id_model, path_to_save)
        _create_zip(path_to_save, path_to_zip)

        return path_to_zip

    def _convert(
            self,
            torch_model: torch.nn.Module,
            new_format: str,
            id_model: str,
            path_to_save: Path
    ) -> None:
        path_to_save_weights = path_to_save / f'{new_format}_{id_model}'
        self._converter[new_format].convert(torch_model, path_to_save_weights)

    @property
    def supported_formats(self) -> list[str]:
        return self._supported_formats

    @property
    def weights_path(self) -> Path:
        return self._weights_path

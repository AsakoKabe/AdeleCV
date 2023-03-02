from __future__ import annotations

import zipfile
from pathlib import Path

from adelecv.api.config import Settings
from adelecv.api.logs import get_logger


class ExportWeights:
    """
    Class for export weights.

    :param weights_path: Path to saved weights.
    """

    def __init__(
            self,
            weights_path: Path = Settings.WEIGHTS_PATH
    ):
        self._weights_path = weights_path

    def create_zip(
            self,
            id_selected: None | set[str] | list[str] = None
    ) -> Path:
        """
        Create zip file with selected models. If no model
         in selected then use all models.

        :param id_selected: List with id models from stats_models
        :return: Path to created zip file.
        """
        zip_path = self._weights_path.parent / 'weights.zip'
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in self._weights_path.rglob("*"):
                if id_selected is None or entry.stem in id_selected:
                    zip_file.write(
                        entry, entry.relative_to(
                            self._weights_path
                        )
                    )
        get_logger().info(
            "Create zip with weights, path: %s", zip_path.as_posix()
        )

        return zip_path

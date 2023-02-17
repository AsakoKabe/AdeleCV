from __future__ import annotations

import zipfile
from pathlib import Path

from config import get_settings


class ExportWeights:
    def __init__(
            self,
            weights_path: Path = get_settings().WEIGHTS_PATH
    ):
        self._weights_path = weights_path

    def create_zip(
            self,
            id_selected: None | set[str] = None
    ) -> Path:
        zip_path = self._weights_path.parent / 'weights.zip'
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for entry in self._weights_path.rglob("*"):
                if id_selected is None or entry.stem in id_selected:
                    zip_file.write(entry, entry.relative_to(self._weights_path))

        return zip_path


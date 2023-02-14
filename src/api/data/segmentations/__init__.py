from .dataset import SegmentationDataset
from .types import COCOSemantic, ImageMask


__all__ = [
    "SegmentationDataset",
    "get_segmentations_dataset_types",
]


_segmentations_dataset_types = [
    COCOSemantic,
    ImageMask,
]


def get_segmentations_dataset_types() -> list[str]:
    return [obj.__name__ for obj in _segmentations_dataset_types]

from .dataset import SegmentationDataset
from .types import COCOSegmentation, ImageMask

__all__ = [
    "SegmentationDataset",
    "get_segmentations_dataset_types",
]


_segmentations_dataset_types = [
    COCOSegmentation,
    ImageMask,
]


def get_segmentations_dataset_types() -> list[str]:
    """
    Get formats of datasets

    :return: List of available dataset formats.
    """
    return [obj.__name__ for obj in _segmentations_dataset_types]

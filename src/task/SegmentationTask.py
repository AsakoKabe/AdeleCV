import itertools
from typing import Tuple, Any, List

from data.dataset.SegmentationDataset import \
    SegmentationDataset
from models.semantic.SegmentationModel import SegmentationModel
from task.base import BaseTask


class SegmentationTask(BaseTask):
    def __init__(self, dataset: SegmentationDataset):
        super().__init__(
            dataset=dataset
        )
        self.models: List[SegmentationModel] = []






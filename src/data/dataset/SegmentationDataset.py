from typing import Tuple, Any

import numpy as np
import torch
from torch.utils.data import DataLoader
import fiftyone as fo
import cv2

from data.dataset.SegmentationTorchDataset import SegmentationTorchDataset
from models.semantic.SegmentationModel import SegmentationModel


class SegmentationDataset:
    def __init__(
            self,
            dataset_dir,
            dataset_type,
            img_size,  # height, width
            split=(0.7, 0.2, 0.1),
            batch_size=16
    ):
        self.dataset_dir = dataset_dir
        self.fo_dataset = dataset_type.create_dataset(self.dataset_dir)
        self.split = split
        self.batch_size = batch_size
        # todo: сделать нормальный resize всего датасета
        self.fo_dataset.info['img_size'] = img_size
        self.fo_dataset.save()
        self.num_classes = len(self.fo_dataset.default_mask_targets)
        self.train, self.val, self.test = None, None, None
        self._transforms = None

    def _split_dataset(self):
        self.fo_dataset.take(
            int(self.split[0] * len(self.fo_dataset))
        ).tag_samples("train")
        self.fo_dataset.match_tags(
            "train",
            bool=False
        ).tag_samples("valid_test")
        self.fo_dataset.match_tags("valid_test").take(
            int(self.split[1] * len(self.fo_dataset))
        ).tag_samples("valid")
        self.fo_dataset.match_tags(
            ["train", "valid"],
            bool=False
        ).tag_samples("test")
        self.fo_dataset.untag_samples('valid_test')

    def _create_torch_datasets(self, transforms):
        train = SegmentationTorchDataset(self.fo_dataset.match_tags('train'), transforms)
        val = SegmentationTorchDataset(self.fo_dataset.match_tags('valid'), transforms)
        test = SegmentationTorchDataset(self.fo_dataset.match_tags('test'), transforms)

        return train, val, test

    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, transforms):
        self._transforms = transforms

    def _create_dataloaders(self) -> None:
        self._split_dataset()

        train_ds, val_ds, test_ds = self._create_torch_datasets(self.transforms)

        self.train = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.val = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=True
        )
        self.test = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

    def update_datasets(self, transforms):
        self.transforms = transforms
        self._create_dataloaders()

    def add_predictions(self, model: SegmentationModel):
        with fo.ProgressBar() as pb:
            for sample in pb(self.fo_dataset.iter_samples(autosave=True)):
                img = cv2.imread(sample.filepath, cv2.IMREAD_COLOR)
                pred = model.predict(img)
                pred = torch.argmax(pred, dim=1).cpu().numpy()[0]
                mask = cv2.resize(
                    np.array(pred, dtype='uint8'),
                    (sample.metadata.width, sample.metadata.height)
                )
                sample[str(model)] = fo.Segmentation(mask=mask)

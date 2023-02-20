import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
import fiftyone as fo
import cv2

from api.models.segmentations import SegmentationModel
from .torch_dataset import SegmentationTorchDataset
from .types import DatasetType
from ...logs import get_logger


class SegmentationDataset:
    def __init__(
            self,
            dataset_dir: str,
            dataset_type: DatasetType,
            img_size: tuple[int, int],  # height, width
            split: tuple[float, float, float] = (0.7, 0.2, 0.1),
            batch_size: int = 16
    ):
        if sum(split) > 1:
            raise ValueError(f"The sum split must be equal to 1, but now {sum(split)}")
        self.dataset_dir = dataset_dir
        self.fo_dataset = dataset_type.create_dataset(self.dataset_dir)
        self.split = split
        if batch_size < 1:
            raise ValueError(f"Butch size must be greater than 0, but now {batch_size}")
        self.batch_size = batch_size
        if len(img_size) < 2:
            raise ValueError(f"Img size must be len = 2 (height, width), but now {len(img_size)}")
        self.img_size = img_size
        self.fo_dataset.save()
        self.num_classes = len(self.fo_dataset.default_mask_targets)
        self.train, self.val, self.test = None, None, None
        self._transforms = None
        self._split_dataset()
        get_logger().info("Creating a dataset")
        get_logger().debug(
            "Dataset created with params, dataset dir: %s, classes: %s, batch size: %s",
            self.dataset_dir, self.fo_dataset.default_mask_targets, self.batch_size
        )

    def _split_dataset(self) -> None:
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

        train_size = len(self.fo_dataset.match_tags(["train"], bool=True))
        valid_size = len(self.fo_dataset.match_tags(["valid"], bool=True))
        test_size = len(self.fo_dataset.match_tags(["test"], bool=True))
        get_logger().info("Split dataset train size: %s, valid size: %s, test size: %s", train_size, valid_size, test_size)

    def _create_torch_datasets(self, transforms: A.Compose) -> tuple[
        SegmentationTorchDataset,
        SegmentationTorchDataset,
        SegmentationTorchDataset
    ]:
        train = SegmentationTorchDataset(self.fo_dataset.match_tags('train'), transforms)
        val = SegmentationTorchDataset(self.fo_dataset.match_tags('valid'), transforms)
        test = SegmentationTorchDataset(self.fo_dataset.match_tags('test'), transforms)

        return train, val, test

    @property
    def transforms(self) -> A.Compose:
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: A.Compose) -> None:
        self._transforms = transforms

    def _create_dataloaders(self) -> None:
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

    def update_datasets(self, transforms: A.Compose) -> None:
        self.transforms = transforms
        self._create_dataloaders()
        get_logger().debug("Dataset updated")

    def add_predictions(self, model: SegmentationModel) -> None:
        get_logger().debug("Add predictions for model: %s", str(model))
        with fo.ProgressBar() as pb:
            for sample in pb(self.fo_dataset.iter_samples(autosave=True)):
                img = cv2.imread(sample.filepath, cv2.IMREAD_COLOR)
                pred = model.predict(img)[0]
                pred = torch.argmax(pred, dim=0).cpu().numpy()
                mask = cv2.resize(
                    np.array(pred, dtype='uint8'),
                    (sample.metadata.width, sample.metadata.height)
                )
                sample[str(model)] = fo.Segmentation(mask=mask)

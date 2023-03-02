import albumentations as A
import cv2
import fiftyone as fo
import numpy as np
import torch
from torch.utils.data import DataLoader

from adelecv.api.logs import get_logger
from adelecv.api.models.segmentations import SegmentationModel

from .torch_dataset import SegmentationTorchDataset
from .types import DatasetType


class SegmentationDataset:
    """
     A class for storing a dataset, its parameters,
     fiftyone sessions and partitioning for training.

    :param dataset_dir: Path to dataset
    :param dataset_type: DatasetType inheritor class, which implements a method
     for converting the output of the dataset format into an internal one
    :param img_size: image size for resize (height, width)
    :param split: percentage of splitting the dataset into train val test
    :param batch_size: batch size for pytorch dataloader
    """

    def __init__(
            self,
            dataset_dir: str,
            dataset_type: DatasetType,
            img_size: tuple[int, int],  # height, width
            split: tuple[float, float, float] = (0.7, 0.2, 0.1),
            batch_size: int = 16
    ):
        if sum(split) > 1:
            raise ValueError(
                f"The sum split must be equal to 1, but now {sum(split)}"
            )
        self._dataset_dir = dataset_dir
        self._fo_dataset = dataset_type.create_dataset(
            self._dataset_dir
        )
        self._split = split
        if batch_size < 1:
            raise ValueError(
                f"Butch size must be greater than 0, but now {batch_size}"
            )
        self._batch_size = batch_size
        if len(img_size) < 2:
            raise ValueError(
                f"Img size must be len = 2 (height, width),"
                f" but now {len(img_size)}"
            )
        self._img_size = img_size
        self._fo_dataset.save()
        self._num_classes = len(self._fo_dataset.default_mask_targets)
        self._train, self._val, self._test = None, None, None
        self._transforms = None
        self._split_dataset()
        get_logger().info("Creating a dataset")
        get_logger().debug(
            "Dataset created with params,"
            " dataset dir: %s, classes: %s, batch size: %s",
            self._dataset_dir, self._fo_dataset.default_mask_targets,
            self._batch_size
        )

    def _split_dataset(self) -> None:
        self._fo_dataset.take(
            int(self._split[0] * len(self._fo_dataset))
        ).tag_samples("train")
        self._fo_dataset.match_tags(
            "train",
            bool=False
        ).tag_samples("valid_test")
        self._fo_dataset.match_tags("valid_test").take(
            int(self._split[1] * len(self._fo_dataset))
        ).tag_samples("valid")
        self._fo_dataset.match_tags(
            ["train", "valid"],
            bool=False
        ).tag_samples("test")
        self._fo_dataset.untag_samples('valid_test')

        train_size = len(self._fo_dataset.match_tags(["train"], bool=True))
        valid_size = len(self._fo_dataset.match_tags(["valid"], bool=True))
        test_size = len(self._fo_dataset.match_tags(["test"], bool=True))
        get_logger().info(
            "Split dataset train size: %s, valid size: %s, test size: %s",
            train_size, valid_size,
            test_size
        )

    def _create_torch_datasets(self, transforms: A.Compose) -> tuple[
        SegmentationTorchDataset,
        SegmentationTorchDataset,
        SegmentationTorchDataset
    ]:
        train = SegmentationTorchDataset(
            self._fo_dataset.match_tags('train'),
            transforms
            )
        val = SegmentationTorchDataset(
            self._fo_dataset.match_tags('valid'),
            transforms
            )
        test = SegmentationTorchDataset(
            self._fo_dataset.match_tags('test'),
            transforms
            )

        return train, val, test

    @property
    def transforms(self) -> A.Compose:
        return self._transforms

    @transforms.setter
    def transforms(self, transforms: A.Compose) -> None:
        self._transforms = transforms

    def _create_dataloaders(self) -> None:
        train_ds, val_ds, test_ds = self._create_torch_datasets(
            self.transforms
        )

        self._train = DataLoader(
            train_ds,
            batch_size=self._batch_size,
            shuffle=True
        )
        self._val = DataLoader(
            val_ds,
            batch_size=self._batch_size,
            shuffle=True
        )
        self._test = DataLoader(
            test_ds,
            batch_size=self._batch_size,
            shuffle=True
        )

    def update_datasets(self, transforms: A.Compose) -> None:
        """
        :meta private:

        Updating is a transformation when creating a new model,
        because each model from the segmentation model pytorch library
         contains its own transformation.

        See: preprocessing smp_

        :param transforms: transforms for image

        .. _smp:
            https://smp.readthedocs.io/en/latest/quickstart.html
        """
        self.transforms = transforms
        self._create_dataloaders()
        get_logger().debug("Dataset updated")

    def add_predictions(self, model: SegmentationModel) -> None:
        """
        :meta private:

        For fiftyone, the dataset adds predicted masks


        :param model: trained SegmentationModel
        """
        get_logger().debug("Add predictions for model: %s", str(model))
        with fo.ProgressBar() as pb:
            for sample in pb(self._fo_dataset.iter_samples(autosave=True)):
                img = cv2.imread(sample.filepath, cv2.IMREAD_COLOR)
                pred = model.predict(img)[0]
                pred = torch.argmax(pred, dim=0).cpu().numpy()
                mask = cv2.resize(
                    np.array(pred, dtype='uint8'),
                    (sample.metadata.width, sample.metadata.height)
                )
                sample[str(model)] = fo.Segmentation(mask=mask)

    @property
    def train(self) -> SegmentationTorchDataset:
        return self._train

    @property
    def val(self) -> SegmentationTorchDataset:
        return self._val

    @property
    def test(self) -> SegmentationTorchDataset:
        return self._test

    @property
    def fo_dataset(self) -> fo.Dataset:
        return self._fo_dataset

    @property
    def img_size(self) -> tuple[int, int]:
        return self._img_size

    @property
    def num_classes(self) -> int:
        return self._num_classes

import os
from pathlib import Path

import fiftyone as fo
import cv2
from fiftyone import dataset_exists, delete_dataset


class DatasetType:
    @staticmethod
    def create_dataset(dataset_dir: str) -> fo.Dataset:
        pass


class COCOSegmentation(DatasetType):
    @staticmethod
    def create_dataset(dataset_dir: str) -> fo.Dataset:
        if dataset_exists(__class__.__name__):
            delete_dataset(__class__.__name__)

        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.COCODetectionDataset,
            name=__class__.__name__
        )
        dataset.delete_sample_field("detections")

        labels = dataset.distinct("segmentations.detections.label")
        map_labels = {i + 1: label for i, label in enumerate(labels)}
        map_labels[0] = 'background'
        dataset.default_mask_targets = map_labels

        for sample in dataset.iter_samples(autosave=True):
            width = sample['metadata']['width']
            height = sample['metadata']['height']
            sample["semantic"] = sample['segmentations'].to_segmentation(
                mask_targets=dataset.default_mask_targets,
                frame_size=(width, height)
            )

        return dataset


class ImageMask(DatasetType):
    @staticmethod
    def create_dataset(dataset_dir: str) -> fo.Dataset:
        samples = []
        path = Path(dataset_dir)
        for filepath in (path / 'image').iterdir():
            img_path = path / 'image' / filepath.name
            mask_path = path / 'mask' / filepath.name

            sample = fo.Sample(filepath=img_path)

            mask = cv2.imread(mask_path.as_posix(), cv2.IMREAD_GRAYSCALE) // 255
            # print(mask.shape)
            sample["semantic"] = fo.Segmentation(mask=mask)
            sample.compute_metadata()
            samples.append(sample)

        dataset = fo.Dataset(__class__.__name__, overwrite=True)
        dataset.add_samples(samples)

        dataset.default_mask_targets = {
            0: 'background',
            1: 'target'
        }

        return dataset

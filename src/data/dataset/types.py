import os

import fiftyone as fo
from skimage.io import imread


class DatasetType:
    @staticmethod
    def create_dataset(dataset_dir):
        pass


class COCOSemantic(DatasetType):
    @staticmethod
    def create_dataset(dataset_dir):
        dataset = fo.Dataset.from_dir(
            dataset_type=fo.types.COCODetectionDataset,
            dataset_dir=dataset_dir,
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


class ImageMaskSemantic(DatasetType):
    @staticmethod
    def create_dataset(dataset_dir):
        samples = []
        for filepath in os.listdir(dataset_dir + '/image'):
            img_path = dataset_dir + '/image/' + filepath
            mask_path = dataset_dir + '/mask/' + filepath

            sample = fo.Sample(filepath=img_path)
            sample.compute_metadata()
            mask = imread(mask_path) // 255
            sample["semantic"] = fo.Segmentation(mask=mask)

            samples.append(sample)

        dataset = fo.Dataset()
        dataset.add_samples(samples)

        dataset.default_mask_targets = {
            0: 'background',
            1: 'target'
        }

        return dataset

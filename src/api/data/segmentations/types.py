import os

import fiftyone as fo
import cv2


class DatasetType:
    @staticmethod
    def create_dataset(dataset_dir: str) -> fo.Dataset:
        pass


class COCOSemantic(DatasetType):
    @staticmethod
    def create_dataset(dataset_dir: str) -> fo.Dataset:
        dataset = fo.Dataset.from_dir(
            dataset_dir=dataset_dir,
            dataset_type=fo.types.COCODetectionDataset,
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
        for filepath in os.listdir(dataset_dir + '/image'):
            img_path = dataset_dir + '/image/' + filepath
            mask_path = dataset_dir + '/mask/' + filepath

            sample = fo.Sample(filepath=img_path)

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) // 255
            # print(mask.shape)
            sample["semantic"] = fo.Segmentation(mask=mask)
            sample.compute_metadata()
            samples.append(sample)

        dataset = fo.Dataset("ImageMask", overwrite=True)
        dataset.add_samples(samples)

        dataset.default_mask_targets = {
            0: 'background',
            1: 'target'
        }

        return dataset

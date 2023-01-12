import fiftyone as fo


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

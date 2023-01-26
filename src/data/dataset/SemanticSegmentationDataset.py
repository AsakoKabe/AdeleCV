from typing import Tuple, Any

import torch
from PIL import Image
from skimage.io import imread
from torch.utils.data import DataLoader
import fiftyone as fo
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2

from data.dataset.SementicTorchDataset import SemanticTorchDataset
from data.dataset.types import DatasetType


class SemanticDataset:
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
        self.train, self.val, self.test = self._create_dataloaders()

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

    def _create_torch_datasets(self):
        train = SemanticTorchDataset(self.fo_dataset.match_tags('train'))
        val = SemanticTorchDataset(self.fo_dataset.match_tags('valid'))
        test = SemanticTorchDataset(self.fo_dataset.match_tags('test'))

        return train, val, test

    def _create_dataloaders(self) -> Tuple[
        DataLoader[Any],
        DataLoader[Any],
        DataLoader[Any]
    ]:
        self._split_dataset()
        train_ds, val_ds, test_ds = self._create_torch_datasets()

        train_dataloader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True
        )
        val_dataloader = DataLoader(
            val_ds,
            batch_size=self.batch_size,
            shuffle=True
        )
        test_dataloader = DataLoader(
            test_ds,
            batch_size=self.batch_size,
            shuffle=True
        )

        return train_dataloader, val_dataloader, test_dataloader

    def add_predictions(self, model):
        transforms = A.Compose([
            A.Resize(*self.fo_dataset.info['img_size']),
            A.Normalize(),
            ToTensorV2()
        ])
        with fo.ProgressBar() as pb:
            for sample in pb(self.fo_dataset.iter_samples(autosave=True)):
                img = transforms(image=imread(sample.filepath))['image'].float().unsqueeze(0)
                pred = model.predict(img.to(model.device))
                pred = torch.argmax(pred, dim=1).cpu().numpy()
                sample[str(model)] = fo.Segmentation(mask=pred)



import torch
from torch.utils.data import Dataset
import cv2
import torch.nn.functional as F
import numpy as np


class SegmentationTorchDataset(Dataset):
    """
    A class to construct a PyTorch segmentations from a FiftyOne segmentations.

    Args: fiftyone_dataset: a FiftyOne segmentations or view that will be used for
    training or testing transforms (None): a list of PyTorch transforms to
    apply to images and targets when loading
    """

    def __init__(
            self,
            fiftyone_dataset,
            transforms
            # augmentations=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.img_paths = self.samples.values("filepath")
        self.classes = self.samples.default_mask_targets.keys()

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = torch.Tensor(sample['semantic']['mask']).long()

        # todo: долго?
        mask = F.one_hot(mask, num_classes=len(self.classes)).numpy()

        # if transforms
        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        return img, mask

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

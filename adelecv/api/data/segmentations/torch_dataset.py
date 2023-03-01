import albumentations as A
import cv2
import fiftyone as fo
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SegmentationTorchDataset(Dataset):
    """
    A class to construct a PyTorch segmentations from a FiftyOne segmentations.
    """

    def __init__(
            self,
            fiftyone_dataset: fo.Dataset,
            transforms: A.Compose
            # augmentations=None,
    ):
        """

        :param fiftyone_dataset: FiftyOne segmentations or view that will
         be used for training or testing
        :param transforms: List of PyTorch transforms to apply to images
         and targets when loading
        """
        self._samples = fiftyone_dataset
        self._transforms = transforms
        if self._transforms is None:
            raise ValueError("Transform must be not None")
        self._img_paths = self._samples.values("filepath")
        self._classes = self._samples.default_mask_targets.keys()

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        img_path = self._img_paths[idx]
        sample = self._samples[img_path]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = torch.Tensor(sample['semantic']['mask']).long()

        mask = F.one_hot(mask, num_classes=len(self._classes)).numpy()

        transformed = self._transforms(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        return img, mask

    def __len__(self) -> int:
        return len(self._img_paths)

    def get_classes(self) -> tuple[str]:
        return self._classes

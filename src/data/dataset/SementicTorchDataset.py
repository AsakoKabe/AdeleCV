import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from skimage.io import imread
import albumentations as A
from torch.utils.data import Dataset
import torch.nn.functional as F


class SemanticTorchDataset(Dataset):
    """
    A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args: fiftyone_dataset: a FiftyOne dataset or view that will be used for
    training or testing transforms (None): a list of PyTorch transforms to
    apply to images and targets when loading
    """

    def __init__(
            self,
            fiftyone_dataset,
            # TODO: aug
            # augmentations=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = A.Compose([
            A.Resize(*self.samples.info['img_size']),
            A.Normalize(),
            ToTensorV2()
        ])
        self.img_paths = self.samples.values("filepath")
        self.labels_map_rev = self.samples.default_mask_targets
        self.classes = self.labels_map_rev.keys()

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        sample = self.samples[img_path]
        img = imread(img_path)
        mask = sample['semantic']['mask']

        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image'].float()
        mask = transformed['mask'].long()
        mask = F.one_hot(mask, num_classes=len(self.classes)).permute(2, 0, 1).float()

        return img, mask

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

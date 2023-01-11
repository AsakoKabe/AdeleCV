import torch
from PIL import Image
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from skimage.io import imread
import albumentations as A


class SemanticTorchDataset(torch.utils.data.Dataset):
    """A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args: fiftyone_dataset: a FiftyOne dataset or view that will be used for
    training or testing transforms (None): a list of PyTorch transforms to
    apply to images and targets when loading
    """

    def __init__(
            self,
            fiftyone_dataset,
            # transforms=None,
    ):
        self.samples = fiftyone_dataset
        # todo: нормализация изображений, возможно уже в моделях
        self.transforms = A.Compose([
            A.Resize(640, 640),
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

        if self.transforms is not None:
            # img, mask = self.transforms(img, mask)
            img = self.transforms(image=img)['image'].float()
            mask = self.transforms(image=mask)['image'].float()

        return img, mask

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

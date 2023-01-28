from torch.utils.data import Dataset
import cv2
import numpy as np


class SegmentationTorchDataset(Dataset):
    """
    A class to construct a PyTorch dataset from a FiftyOne dataset.

    Args: fiftyone_dataset: a FiftyOne dataset or view that will be used for
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
        mask = sample['semantic']['mask']

        masks = [(mask == v) for v in self.classes]
        mask = np.stack(masks, axis=-1).astype('float')

        transformed = self.transforms(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        return img, mask

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes

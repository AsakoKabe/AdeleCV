from abc import ABC

import torch
from skimage.transform import resize
from torch.utils.data import Dataset
from skimage.io import imread


class BaseDataset(ABC):
    pass
    # дополнительные методы которые будут нужны


class CocoDataset(Dataset):
    def __init__(self, imgs_path, mask_path):
        self.imgs_path = imgs_path
        self.masks_path = mask_path

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        mask_path = self.masks_path[idx]
        img = imread(img_path)
        mask = imread(mask_path)
        # todo: выкинуть
        size = (256, 256)
        img = resize(img, size, mode='constant', anti_aliasing=True,)
        mask = resize(mask, size, mode='constant', anti_aliasing=False) > 0.5
        img = torch.Tensor(img).permute(2, 0, 1)
        mask = torch.Tensor(mask)
        return img, mask

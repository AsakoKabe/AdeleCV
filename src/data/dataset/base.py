from abc import ABC

from torch.utils.data import Dataset
import cv2 as cv


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
        img = cv.imread(img_path, mode='RGB')
        mask = cv.imread(mask_path, mode='RGB')

        return img, mask

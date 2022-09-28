import numpy as np
import cv2 as cv

import os

from base import BaseDataset


class SemanticSegmentationDataset(BaseDataset):
    def __init__(self, img_folder_path, mask_folder_path):
        self.img_folder_path = img_folder_path
        self.mask_folder_path = mask_folder_path
        self.imgs_path = np.array(os.listdir(self.img_folder_path))
        self.masks_path = self.imgs_path

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, idx):
        img_path = self.imgs_path[idx]
        mask_path = self.masks_path[idx]
        img = cv.imread(img_path, mode='RGB')
        mask = cv.imread(mask_path, mode='RGB')

        return img, mask

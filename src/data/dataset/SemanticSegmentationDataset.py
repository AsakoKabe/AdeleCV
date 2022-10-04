import numpy as np
import cv2 as cv

import os

from base import BaseDataset, CocoDataset


class SemanticSegmentationDataset(BaseDataset):
    def __init__(self, path_to_dataset, split=(0.7, 0.2, 0.1)):
        self.img_folder_path = path_to_dataset + '/image'
        self.mask_folder_path = path_to_dataset + '/mask'
        self.imgs_path = np.array(os.listdir(self.img_folder_path))
        self.masks_path = self.imgs_path
        self.split = [round(size * len(self.imgs_path)) for size in split]
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.batch_size = 16

    def __split_dataset(self):
        ix = np.random.choice(len(self.imgs_path), len(self.imgs_path), False)
        train, val, test = np.split(ix, [self.split[0], self.split[1]])

        return (self.imgs_path[train], self.masks_path[train]),\
               (self.imgs_path[val], self.masks_path[val]),\
               (self.imgs_path[test], self.masks_path[test])

    def __create_datasets(self):
        train, val, test = self.__split_dataset()
        self.train_dataset = CocoDataset(*train)
        self.val_dataset = CocoDataset(*val)
        self.test_dataset = CocoDataset(*val)

import numpy as np
import pandas as pd
import torchvision.transforms.functional as F
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.utils import _log_api_usage_once

import constants.columns as cc

class FacialKeypointsDataset(Dataset):
    # Define slots (for save concrete attributes of class in memory)
    __slots__ = ['df', 'transforms', 'images', 'keypoints']

    # Define the constructor with DataFrame and transforms parameters
    def __init__(self, dataframe: pd.DataFrame, transforms=None):
        self.df = dataframe
        self.transforms = transforms

        self.images = []
        self.keypoints = []

        for index, row in self.df.iterrows():
            # Image
            image = row[cc.COLUMN_IMAGE]
            image = np.fromstring(image, sep=' ').reshape([96, 96])

            # Add three equal images to stack (on last axis)
            image = np.stack((image, image, image), axis=-1)
            # Normalize images
            image = image / 255.0

            # Keypoints
            keypoints = row.drop([cc.COLUMN_IMAGE])
            keypoints = keypoints.to_numpy().astype('float32')

            # Add to Dataset's images and keypoints
            self.images.append(image)
            self.keypoints.append(keypoints)

    def __getitem__(self, idx):
        image = self.images[idx]
        keypoints = self.keypoints[idx]

        return_dict = {
            'image': image,
            'keypoints': keypoints
        }

        if self.transforms:
            return_dict = self.transforms(return_dict)

        return return_dict

    def __len__(self):
        return len(self.df)

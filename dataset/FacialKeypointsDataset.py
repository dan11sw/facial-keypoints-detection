import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

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

        # Prepare all elements of dataframe using iter rows method
        for index, row in tqdm(self.df.iterrows(), desc="Processing load dataset", total=len(self.df)):
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

        print("\n✅ Load dataset completed!")

    def __getitem__(self, idx):
        image = self.images[idx]
        keypoints = self.keypoints[idx]

        return_dict = {
            cc.COLUMN_image: image,
            cc.COLUMN_keypoint: keypoints
        }

        if self.transforms:
            return_dict = self.transforms(return_dict)

        return return_dict

    def __len__(self):
        return len(self.df)

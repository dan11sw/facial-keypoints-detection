import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import cv2
import random
import copy
import torchvision.transforms.functional as F
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.utils import _log_api_usage_once
import warnings
import os

from data_preprocessing.extract_data import extract_files
import utils.c_utils as c_utils
import utils.v_utils as v_utils
import constants.columns as cc

warnings.filterwarnings("ignore")
DATA_FILEPATH = "./data"

def print_source_dir(dir=DATA_FILEPATH):
    for dir_name, _, filenames in os.walk(dir):
        for filename in filenames:
            print(os.path.join(dir_name, filename))

#print_source_dir()


torch.manual_seed(2)
np.random.seed(3)
random.seed(4)

# Extract training.csv and test.csv, if these not exists
extract_files(DATA_FILEPATH)

# Load data
train_data = pd.read_csv(DATA_FILEPATH + "/training.csv")
test_data = pd.read_csv(DATA_FILEPATH + "/test.csv")
id_lookup_data = pd.read_csv(DATA_FILEPATH + "/IdLookupTable.csv")

def print_data_info():
    print("Shape loaded data: ", train_data.shape, test_data.shape, id_lookup_data.shape)
    print("Train data head (3):")
    print(train_data.head(3))
    print()
    print("Test data head (3): ")
    print(test_data.head(3))
    print()
    print(train_data.info())
    print()
    print(test_data.info())
    print()

#print_data_info()

# Visualization of NaN elements
def print_visual_nan(data=train_data):
    plt.subplots(figsize=(15, 12))
    sns.heatmap(data.isna().T, cbar=False, cmap='viridis')
    plt.show()

#print_visual_nan()

# Example show image with keypoints by id in train_data
#c_utils.show_image_with_keypoints_by_id(train_data, 20)


# Grouping by duplicates
duplicates = train_data.duplicated(subset=cc.COLUMN_IMAGE, keep='last')
duplicates_list = [g for _, g in train_data.groupby(cc.COLUMN_IMAGE) if len(g) > 1]

print(f"There are {sum(duplicates)} duplicated")
print(len(duplicates_list))

duplicates_df = pd.concat(duplicates_list)
print(duplicates_df.info())
print(duplicates_df.shape)

# Example show images
#v_utils.show_count_images(duplicates_list)

# Delete duplicates
c_utils.delete_duplicates(train_data, duplicates_list)
del duplicates, duplicates_df, duplicates_list

# Reset index
train_data.reset_index(drop=True, inplace=True)

vals = [1, 2, torch.nan, 4]
tens = torch.tensor(vals)

print(~torch.isnan(tens))

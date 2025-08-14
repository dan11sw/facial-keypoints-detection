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

from dataset.FacialKeypointsDataset import FacialKeypointsDataset
from dataset.CustomRandomRotation import CustomRandomRotation
from dataset.CustomRandomTranslation import CustomRandomTranslation
from dataset.CustomRandomVerticalFlip import CustomRandomVerticalFlip
from dataset.CustomRandomBrightnessAdjust import CustomRandomBrightnessAdjust
from dataset.CustomToTensor import CustomToTensor

warnings.filterwarnings("ignore")
DATA_FILEPATH = "./data"

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Choose CUDA device
device = v_utils.cuda_device_info()


def print_source_dir(ex_dir_name=DATA_FILEPATH):
    for dir_name, _, filenames in os.walk(ex_dir_name):
        for filename in filenames:
            print(os.path.join(dir_name, filename))


# print_source_dir()

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


# print_data_info()

# Visualization of NaN elements
def print_visual_nan(data=train_data):
    plt.subplots(figsize=(15, 12))
    sns.heatmap(data.isna().T, cbar=False, cmap='viridis')
    plt.show()


# print_visual_nan()

# Example show image with keypoints by id in train_data
# c_utils.show_image_with_keypoints_by_id(train_data, 20)


# Grouping by duplicates
duplicates = train_data.duplicated(subset=cc.COLUMN_IMAGE, keep='last')
duplicates_list = [g for _, g in train_data.groupby(cc.COLUMN_IMAGE) if len(g) > 1]

print(f"There are {sum(duplicates)} duplicated")
print(len(duplicates_list))

duplicates_df = pd.concat(duplicates_list)
print(duplicates_df.info())
print(duplicates_df.shape)

# Example show images
# v_utils.show_count_images(duplicates_list)

# Delete duplicates
c_utils.delete_duplicates(train_data, duplicates_list)
del duplicates, duplicates_df, duplicates_list

# Reset index
train_data.reset_index(drop=True, inplace=True)

# Load dataset with FacialKeypointsDataset
dataset = FacialKeypointsDataset(
    pd.read_csv(DATA_FILEPATH + "/training.csv")
)

datapoint = dataset[0]
image, keypoints = datapoint["image"], datapoint["keypoints"]


def data_augment_test():
    random_flip = CustomRandomVerticalFlip(p=1.0)
    random_rotation = CustomRandomRotation(p=1.0, angle=45)
    random_translation = CustomRandomTranslation(translate=(0.1, 0.1), p=1.0)
    random_brightness = CustomRandomBrightnessAdjust(brightness=0.2, p=1.0)

    transform_flip = random_flip(datapoint)
    transform_rotation = random_rotation(datapoint)
    transform_translation = random_translation(datapoint)
    transform_combined = random_flip((random_rotation(random_translation(datapoint))))
    transform_brightness = random_brightness(datapoint)

    transforms_list = [transform_flip, transform_rotation,
                       transform_translation, transform_combined,
                       transform_brightness]

    v_utils.show_augment_images(image, keypoints, transforms_list)


# Example show augmented images
# data_augment_test()

BATCH_SIZE = 64
EPOCHS = 120
EPOCHS_PRETRAIN = 25

# Create model and her transfer to GPU
model = torchvision.models.resnet18(pretrained=True)
model.fc = torch.nn.Linear(model.fc.in_features, 30)

model = model.type(torch.FloatTensor)
model = model.to(device)

# Create conveyor for augmented training dataset
train_transform = torchvision.transforms.Compose([
    CustomRandomVerticalFlip(p=0.5),
    CustomRandomRotation(angle=35, p=0.33),
    CustomRandomTranslation(translate=(0.12, 0.12), p=0.33),
    CustomRandomBrightnessAdjust(brightness=0.2, p=0.5),
    CustomToTensor()
])

val_transform = torchvision.transforms.Compose([CustomToTensor()])

# Dataset
train_size = int(len(dataset) * 0.85)
val_size = len(dataset) - train_size

# Random split data
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_set.dataset.transforms = train_transform
val_set.dataset.transforms = val_transform

train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=BATCH_SIZE)
val_loader = torch.utils.data.DataLoader(val_set, shuffle=True, batch_size=BATCH_SIZE)

# Parameters
*previous_layers, last_layer = model.parameters()
print(f"Previous layers: {len(previous_layers)}")

for layer in previous_layers:
    layer.requires_grad = False

# Config optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=0.001)

logger = {
    'train': [],
    'val': []
}

train_steps = len(train_set) / BATCH_SIZE
val_steps = len(val_set) / BATCH_SIZE

for epoch in range(EPOCHS_PRETRAIN):
    train_loss = 0.0
    torch.manual_seed(1 + epoch)

    print(f"EPOCH: {epoch + 1} / {EPOCHS_PRETRAIN}")
    model.train()

    for (batch_idx, sample) in enumerate(train_loader):
        x = sample[cc.COLUMN_image]
        y = sample[cc.COLUMN_keypoint]

        (x, y) = (x.to(device), y.to(device))

        pred = model(x)
        loss = c_utils.NaNMSELoss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

    with torch.no_grad():
        model.eval()
        val_loss = 0.0

        for val_sample in val_loader:
            x = val_sample[cc.COLUMN_image]
            y = val_sample[cc.COLUMN_keypoint]

            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            val_loss += c_utils.NaNMSELoss(pred, y)

    avg_train_loss = train_loss / val_steps
    avg_val_loss = val_loss / val_steps

    logger["train"].append(avg_train_loss.cpu().detach().numpy())
    logger["val"].append(avg_val_loss.cpu().detach().numpy())

    print(f"Average train loss: {avg_train_loss:.6f}, Average validation loss: {avg_val_loss:.6f}")

logger_df = pd.DataFrame(logger)
logger_df.to_csv("logger_resnet18.csv")

# Save learned model
torch.save(model.state_dict(), "model_resnet18.pth")

for layer in previous_layers:
    layer.requires_grad = True

optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.00005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                 milestones=[5, 10, 25, 40, 65],
                                                 gamma=0.1)

logger = {'train': [], 'val': []}

best_model = None
min_val_loss = np.inf

train_steps = len(train_set) / BATCH_SIZE
val_steps = len(val_set) / BATCH_SIZE

for epoch in range(EPOCHS):
    torch.manual_seed(1 + epoch)

    print(f"EPOCH: {epoch + 1} / {EPOCHS}")

    model.train()
    train_loss = 0.0

    for (batch_idx, sample) in enumerate(train_loader):
        x = sample[cc.COLUMN_image]
        y = sample[cc.COLUMN_keypoint]

        (x, y) = (x.to(device), y.to(device))

        pred = model(x)
        loss = c_utils.NaNMSELoss(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss

    with torch.no_grad():
        model.eval()
        val_loss = 0.0

        for val_sample in val_loader:
            x = val_sample[cc.COLUMN_image]
            y = val_sample[cc.COLUMN_keypoint]

            (x, y) = (x.to(device), y.to(device))

            pred = model(x)
            val_loss += c_utils.NaNMSELoss(pred, y)

    scheduler.step()

    avg_train_loss = train_loss / val_steps
    avg_val_loss = val_loss / val_steps

    logger["train"].append(avg_train_loss.cpu().detach().numpy())
    logger["val"].append(avg_val_loss.cpu().detach().numpy())

    print(f"Average train loss: {avg_train_loss:.6f}, Average validation loss: {avg_val_loss:.6f}")

    if min_val_loss > val_loss:
        min_test_loss = val_loss
        # Deepcopy model
        best_model = copy.deepcopy(model)

logger_df = pd.DataFrame(logger)
logger_df.to_csv('logger_custom.csv')

# Save learned model
torch.save(best_model.state_dict(), "model_custom.pth")

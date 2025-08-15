import numpy as np
import torch
import torchvision
import pandas as pd
import copy

from dataset.CustomRandomRotation import CustomRandomRotation
from dataset.CustomRandomTranslation import CustomRandomTranslation
from dataset.CustomRandomVerticalFlip import CustomRandomVerticalFlip
from dataset.CustomRandomBrightnessAdjust import CustomRandomBrightnessAdjust
from dataset.CustomToTensor import CustomToTensor
import constants.columns as cc
import utils.c_utils as c_utils


class CustomResNet18(torch.nn.Module):
    def __init__(self, num_classes=30, pretrained=True):
        super().__init__()

        # Load pretrained ResNet18 model
        self.backbone = torchvision.models.resnet18(pretrained=pretrained)
        self.backbone.fc = torch.nn.Linear(self.backbone.fc.in_features, num_classes)

    def freeze_layers(self, freeze=False):
        # Parameters
        # *previous_layers, last_layer = self.backbone.parameters()
        # print(f"Previous layers: {len(previous_layers)}")

        for layer in self.backbone.parameters():
            layer.requires_grad = not freeze

        for layer in self.backbone.fc.parameters():
            layer.requires_grad = True

    def forward(self, x):
        return self.backbone(x)


class CustomDatasetPipeline:
    __slots__ = [
        'batch_size', 'dataset', 'train_transform',
        'val_transform', 'train_set', 'val_set'
    ]

    def __init__(self, dataset, batch_size=32):
        self.batch_size = batch_size
        self.dataset = dataset

        # Create conveyor for augmented training dataset
        self.train_transform = torchvision.transforms.Compose([
            CustomRandomVerticalFlip(p=0.5),
            CustomRandomRotation(angle=35, p=0.33),
            CustomRandomTranslation(translate=(0.12, 0.12), p=0.33),
            CustomRandomBrightnessAdjust(brightness=0.2, p=0.5),
            CustomToTensor()
        ])

        self.val_transform = torchvision.transforms.Compose([CustomToTensor()])

        # Split train data
        self._split_data()

    def _split_data(self):
        train_size = int(len(self.dataset) * 0.85)
        val_size = int(len(self.dataset)) - train_size

        self.train_set, self.val_set = torch.utils.data.random_split(
            self.dataset, [train_size, val_size]
        )

        # Used transformations
        self.train_set.dataset.transforms = self.train_transform
        self.val_set.dataset.transforms = self.val_transform

    def get_loaders(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False
        )

        return train_loader, val_loader


class CustomTrainer:
    __slots__ = [
        "model", "train_loader", "val_loader",
        "device", "optimizer", "train_loss",
        "val_loss", "logger", "scheduler",
        "min_val_loss", "best_model"
    ]

    def __init__(self, model, train_loader, val_loader, device, min_val_loss=np.inf):
        self.model = model.to(device)
        self.best_model = None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.1,
            weight_decay=0.001
        )

        self.train_loss = 0.0
        self.val_loss = 0.0
        self.min_val_loss = min_val_loss

        self.logger = {
            'train': [],
            'val': []
        }

        self.scheduler = None

    def reset_logger(self):
        self.logger = {
            'train': [],
            'val': []
        }

    def train_epoch(self):
        self.model.train()
        value_loss = 0.0

        for (batch_idx, sample) in enumerate(self.train_loader):
            x = sample[cc.COLUMN_image].to(self.device)
            y = sample[cc.COLUMN_keypoint].to(self.device)

            self.optimizer.zero_grad()
            predict = self.model(x)
            loss = c_utils.NaNMSELoss(predict, y)
            loss.backward()
            self.optimizer.step()

            value_loss += loss.item()

        return value_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        value_loss = 0.0

        with torch.no_grad():
            for val_sample in self.val_loader:
                x = val_sample[cc.COLUMN_image].to(self.device)
                y = val_sample[cc.COLUMN_keypoint].to(self.device)

                predict = self.model(x)
                value_loss += c_utils.NaNMSELoss(predict, y).item()

        if self.scheduler is not None:
            self.scheduler.step()

        return value_loss / len(self.val_loader)

    def train(self, epochs, title=None):
        if title is not None:
            print(title + "\n")

        self.train_loss = 0.0
        self.val_loss = 0.0

        for epoch in range(epochs):
            torch.manual_seed(1 + epoch)
            print(f"EPOCH: {epoch + 1} / {epochs}")

            self.train_loss = self.train_epoch()
            self.val_loss = self.validate()

            self.logger['train'].append(self.train_loss)
            self.logger['val'].append(self.val_loss)

            print(f"Train loss: {self.train_loss:.6f}, Val loss: {self.val_loss:.6f}\n")

            if self.min_val_loss > self.val_loss:
                self.min_val_loss = self.val_loss
                self.best_model = copy.deepcopy(self.model)

    def save_results(self, model_path="model.pth", log_path="logger.csv"):
        if self.best_model is not None:
            torch.save(self.best_model.state_dict(), model_path)
        else:
            torch.save(self.model.state_dict(), model_path)

        pd.DataFrame(self.logger).to_csv(log_path)

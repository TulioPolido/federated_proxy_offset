import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flautim.pytorch.Dataset import Dataset


class RecurrenceDataset(Dataset):
    def __init__(
        self,
        train_file=None,
        test_file=None,
        use_rotation=False,
        support_ratio=0.4,
        stratified=True,
        seed=42,
        **kwargs,
    ):
        super(RecurrenceDataset, self).__init__(name="RecurrenceDataset", **kwargs)

        # =========================
        # Load train data
        # =========================

        self.data = []
        self.labels = []
        # =========================
        # Load test data
        # =========================
        self.test_data = []
        self.test_labels = []

        # =========================
        # Dataloader parameters
        # =========================
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.num_workers = kwargs.get("num_workers", 1)

    # ==========================================================
    # Dataset API
    # ==========================================================
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.train_mode and self.use_rotation:
            img = self._augment_image(img.clone())

        return img, label

    # ==========================================================
    # Train / Validation copies (deepcopy as requested)
    # ==========================================================
    def train(self):
        train = copy.deepcopy(self)
        train.data = self.data
        train.labels = self.labels
        train.train_mode = True
        return train

    def validation(self):
        val = copy.deepcopy(self)
        val.data = self.test_data
        val.labels = self.test_labels
        val.train_mode = False
        return val

    # ==========================================================
    # Meta-compatible dataloader
    # ==========================================================
    def dataloader(self, validation=False):
        """
        validation=False → returns [train_support_loader, train_query_loader]
        validation=True  → returns [test_support_loader, test_query_loader]
        """

        return []
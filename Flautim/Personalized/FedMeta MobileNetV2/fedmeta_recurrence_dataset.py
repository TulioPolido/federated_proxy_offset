import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from flautim.pytorch.Dataset import Dataset


class RecurrenceDataset(Dataset):
    def __init__(
        self,
        train_file,
        test_file=None,
        use_rotation=False,
        support_ratio=0.2,
        stratified=True,
        seed=42,
        **kwargs,
    ):
        super(RecurrenceDataset, self).__init__(name="RecurrenceDataset", **kwargs)

        # =========================
        # Load train data
        # =========================
        self.data = train_file["data"].float().permute(0, 3, 1, 2)
        self.labels = train_file["labels"].long()

        # =========================
        # Load test data
        # =========================
        self.test_data = test_file["data"].float().permute(0, 3, 1, 2)
        self.test_labels = test_file["labels"].long()

        # =========================
        # Dataloader parameters
        # =========================
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.num_workers = kwargs.get("num_workers", 1)

        # =========================
        # Meta-learning parameters
        # =========================
        self.support_ratio = support_ratio
        self.stratified = stratified
        self.seed = seed

        # =========================
        # Augmentation
        # =========================
        self.use_rotation = use_rotation
        self.train_mode = True
        self.angles = [0, 90, 180, 270]

        # =========================
        # Create FIXED support/query splits
        # =========================
        rng = np.random.RandomState(self.seed)

        self.train_support_idx, self.train_query_idx = self._split_support_query(
            self.labels, rng
        )

        self.test_support_idx, self.test_query_idx = self._split_support_query(
            self.test_labels, rng
        )

    # ==========================================================
    # Support / Query split (deterministic)
    # ==========================================================
    def _split_support_query(self, labels_tensor, rng):
        labels_np = labels_tensor.cpu().numpy()
        indices = np.arange(len(labels_np))

        support_indices = []
        query_indices = []

        if self.stratified:
            for label in np.unique(labels_np):
                label_idx = indices[labels_np == label]
                rng.shuffle(label_idx)

                split = max(1, int(len(label_idx) * self.support_ratio))
                support_indices.extend(label_idx[:split])
                query_indices.extend(label_idx[split:])
        else:
            rng.shuffle(indices)
            split = max(1, int(len(indices) * self.support_ratio))
            support_indices = indices[:split]
            query_indices = indices[split:]

        return support_indices, query_indices

    # ==========================================================
    # Augmentation
    # ==========================================================
    def _augment_image(self, img_tensor):
        if self.use_rotation:
            angle = random.choice(self.angles)
            k = angle // 90
            img_tensor = torch.rot90(img_tensor, k=k, dims=(1, 2))
        return img_tensor

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
    # Train / Validation copies
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

        if validation:
            dataset = self.validation()
            support_idx = self.test_support_idx
            query_idx = self.test_query_idx
        else:
            dataset = self.train()
            support_idx = self.train_support_idx
            query_idx = self.train_query_idx

        support_set = Subset(dataset, support_idx)
        query_set = Subset(dataset, query_idx)

        support_loader = DataLoader(
            support_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        query_loader = DataLoader(
            query_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return [support_loader, query_loader]

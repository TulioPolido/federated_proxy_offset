import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from flautim.pytorch.Dataset import Dataset


class RecurrenceDatasetLOSO(Dataset):
    def __init__(
        self,
        train_file,
        test_file=None,
        use_rotation=False,
        support_ratio=0.4,
        stratified=True,
        seed=42,
        **kwargs,
    ):
        super(RecurrenceDatasetLOSO, self).__init__(name="RecurrenceDatasetLOSO", **kwargs)

        # =========================
        # Load train data
        # =========================
        train_x = train_file["data"].float().permute(0, 3, 1, 2)
        train_y = train_file["labels"].long()

        # =========================
        # Load test data
        # =========================
        val_x = test_file["data"].float().permute(0, 3, 1, 2)
        val_y = test_file["labels"].long()

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

        rng = np.random.RandomState(self.seed)

        # ==================================================
        # Split TRAIN → support/query
        # ==================================================
        train_sup_idx, train_qry_idx = self._split_support_query(train_y, rng)

        self.train_sup_x = train_x[train_sup_idx]
        self.train_sup_y = train_y[train_sup_idx]

        self.train_qry_x = train_x[train_qry_idx]
        self.train_qry_y = train_y[train_qry_idx]

        # ==================================================
        # Split TEST → support/query
        # ==================================================
        val_sup_idx, val_qry_idx = self._split_support_query(val_y, rng)

        self.val_sup_x = val_x[val_sup_idx]
        self.val_sup_y = val_y[val_sup_idx]

        self.val_qry_x = val_x[val_qry_idx]
        self.val_qry_y = val_y[val_qry_idx]

        # Default dataset content (used by __getitem__)
        self.data = self.train_sup_x
        self.labels = self.train_sup_y

    # ==========================================================
    # Support / Query split
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

                split = int(len(label_idx) * self.support_ratio)

                support_indices.extend(label_idx[:split])
                query_indices.extend(label_idx[split:])
        else:
            rng.shuffle(indices)
            split = int(len(indices) * self.support_ratio)

            support_indices = indices[:split]
            query_indices = indices[split:]

        return np.array(support_indices), np.array(query_indices)

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
    # Meta-compatible dataloader
    # ==========================================================
    def dataloader(self, validation=False):

        if validation:
            sup_x, sup_y = self.val_sup_x, self.val_sup_y
            qry_x, qry_y = self.val_qry_x, self.val_qry_y
        else:
            sup_x, sup_y = self.train_sup_x, self.train_sup_y
            qry_x, qry_y = self.train_qry_x, self.train_qry_y

        support_dataset = copy.deepcopy(self)
        support_dataset.data = sup_x
        support_dataset.labels = sup_y
        support_dataset.train_mode = not validation

        query_dataset = copy.deepcopy(self)
        query_dataset.data = qry_x
        query_dataset.labels = qry_y
        query_dataset.train_mode = False

        support_loader = DataLoader(
            support_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return [support_loader, query_loader]

class RecurrenceDataset(Dataset):
    def __init__(
        self,
        dataset_file,
        train_ratio=0.8,
        support_ratio=0.4,
        stratified=True,
        seed=42,
        use_rotation=False,
        **kwargs,
    ):
        super(RecurrenceDataset, self).__init__(name="RecurrenceDataset", **kwargs)

        # =========================
        # Load full dataset
        # =========================
        data = dataset_file["data"].float().permute(0, 3, 1, 2)
        labels = dataset_file["labels"].long()

        # =========================
        # Dataloader parameters
        # =========================
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.num_workers = kwargs.get("num_workers", 1)

        # =========================
        # Parameters
        # =========================
        self.train_ratio = train_ratio
        self.support_ratio = support_ratio
        self.stratified = stratified
        self.seed = seed

        # =========================
        # Augmentation
        # =========================
        self.use_rotation = use_rotation
        self.train_mode = True
        self.angles = [0, 90, 180, 270]

        rng = np.random.RandomState(seed)

        # ==================================================
        # 1️⃣ TRAIN / VALIDATION SPLIT
        # ==================================================
        train_idx, val_idx = self._split_train_val(labels, rng)

        train_x = data[train_idx]
        train_y = labels[train_idx]

        val_x = data[val_idx]
        val_y = labels[val_idx]

        # ==================================================
        # 2️⃣ SUPPORT / QUERY SPLIT (TRAIN)
        # ==================================================
        train_sup_idx, train_qry_idx = self._split_support_query(train_y, rng)

        self.train_sup_x = train_x[train_sup_idx]
        self.train_sup_y = train_y[train_sup_idx]

        self.train_qry_x = train_x[train_qry_idx]
        self.train_qry_y = train_y[train_qry_idx]

        # ==================================================
        # 3️⃣ SUPPORT / QUERY SPLIT (VALIDATION)
        # ==================================================
        val_sup_idx, val_qry_idx = self._split_support_query(val_y, rng)

        self.val_sup_x = val_x[val_sup_idx]
        self.val_sup_y = val_y[val_sup_idx]

        self.val_qry_x = val_x[val_qry_idx]
        self.val_qry_y = val_y[val_qry_idx]

    # ==================================================
    # Train / Validation split
    # ==================================================
    def _split_train_val(self, labels, rng):
        labels_np = labels.cpu().numpy()
        indices = np.arange(len(labels_np))

        if self.stratified:
            train_idx = []
            val_idx = []

            for label in np.unique(labels_np):
                label_indices = indices[labels_np == label]
                rng.shuffle(label_indices)

                split = int(len(label_indices) * self.train_ratio)

                train_idx.extend(label_indices[:split])
                val_idx.extend(label_indices[split:])
        else:
            rng.shuffle(indices)
            split = int(len(indices) * self.train_ratio)

            train_idx = indices[:split]
            val_idx = indices[split:]

        return np.array(train_idx), np.array(val_idx)

    # ==================================================
    # Support / Query split
    # ==================================================
    def _split_support_query(self, labels, rng):
        labels_np = labels.cpu().numpy()
        indices = np.arange(len(labels_np))

        support_indices = []
        query_indices = []

        if self.stratified:
            for label in np.unique(labels_np):
                label_idx = indices[labels_np == label]
                rng.shuffle(label_idx)

                split = int(len(label_idx) * self.support_ratio)

                support_indices.extend(label_idx[:split])
                query_indices.extend(label_idx[split:])
        else:
            rng.shuffle(indices)
            split = int(len(indices) * self.support_ratio)

            support_indices = indices[:split]
            query_indices = indices[split:]

        return np.array(support_indices), np.array(query_indices)

    # ==================================================
    # Augmentation
    # ==================================================
    def _augment_image(self, img_tensor):
        if self.use_rotation:
            angle = random.choice(self.angles)
            k = angle // 90
            img_tensor = torch.rot90(img_tensor, k=k, dims=(1, 2))
        return img_tensor

    # ==================================================
    # Dataloader
    # ==================================================
    
    def dataloader(self, validation=False):

        if validation:
            sup_x, sup_y = self.val_sup_x, self.val_sup_y
            qry_x, qry_y = self.val_qry_x, self.val_qry_y
        else:
            sup_x, sup_y = self.train_sup_x, self.train_sup_y
            qry_x, qry_y = self.train_qry_x, self.train_qry_y

        # Create dataset copies
        support_dataset = copy.deepcopy(self)
        support_dataset.data = sup_x
        support_dataset.labels = sup_y

        query_dataset = copy.deepcopy(self)
        query_dataset.data = qry_x
        query_dataset.labels = qry_y
        query_dataset.train_mode = False

        support_loader = DataLoader(
            support_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        return [support_loader, query_loader]
import copy
import random

import numpy as np
import torch
from torch.utils.data import DataLoader
from flautim.pytorch.Dataset import Dataset


class TimeSeriesDatasetLOSO(Dataset):

    def __init__(
        self,
        train_file,
        test_file=None,
        support_ratio=0.3,
        stratified=True,
        seed=42,
        **kwargs,
    ):

        super(TimeSeriesDatasetLOSO, self).__init__(
            name="TimeSeriesDatasetLOSO",
            **kwargs
        )

        # ==================================================
        # Load train data
        # ==================================================

        # Keep the time-series representation.
        #
        # Expected:
        # [N, 1024, 6]
        #
        # Do NOT permute here.
        train_x = train_file["data"].float()
        train_y = train_file["labels"].long()

        # ==================================================
        # Load test data
        # ==================================================

        val_x = test_file["data"].float()
        val_y = test_file["labels"].long()

        # ==================================================
        # Validate shapes
        # ==================================================

        if train_x.ndim != 3:
            raise ValueError(
                f"Expected train data to have 3 dimensions, "
                f"got {train_x.shape}"
            )

        if val_x.ndim != 3:
            raise ValueError(
                f"Expected test data to have 3 dimensions, "
                f"got {val_x.shape}"
            )

        # Accept either:
        #
        # [N, 1024, 6]
        # or
        # [N, 6, 1024]
        #
        # Internally we keep [N, 1024, 6].

        if train_x.shape[-1] == 6:
            pass

        elif train_x.shape[1] == 6:
            train_x = train_x.transpose(1, 2)

        else:
            raise ValueError(
                f"Expected train data shape [N, 1024, 6] "
                f"or [N, 6, 1024], got {train_x.shape}"
            )

        if val_x.shape[-1] == 6:
            pass

        elif val_x.shape[1] == 6:
            val_x = val_x.transpose(1, 2)

        else:
            raise ValueError(
                f"Expected test data shape [N, 1024, 6] "
                f"or [N, 6, 1024], got {val_x.shape}"
            )

        # ==================================================
        # Dataloader parameters
        # ==================================================

        self.batch_size = kwargs.get(
            "batch_size",
            32
        )

        self.shuffle = kwargs.get(
            "shuffle",
            True
        )

        self.num_workers = kwargs.get(
            "num_workers",
            1
        )

        # ==================================================
        # Meta-learning parameters
        # ==================================================

        self.support_ratio = support_ratio
        self.stratified = stratified
        self.seed = seed

        # ==================================================
        # No image augmentation
        # ==================================================

        self.train_mode = True

        # ==================================================
        # Train -> Support / Query
        # ==================================================

        rng = np.random.RandomState(
            self.seed
        )

        train_sup_idx, train_qry_idx = (
            self._split_support_query(
                train_y,
                rng
            )
        )

        self.train_sup_x = train_x[
            train_sup_idx
        ]

        self.train_sup_y = train_y[
            train_sup_idx
        ]

        self.train_qry_x = train_x[
            train_qry_idx
        ]

        self.train_qry_y = train_y[
            train_qry_idx
        ]

        # ==================================================
        # Test -> Support / Query
        # ==================================================

        val_sup_idx, val_qry_idx = (
            self._split_support_query(
                val_y,
                rng
            )
        )

        self.val_sup_x = val_x[
            val_sup_idx
        ]

        self.val_sup_y = val_y[
            val_sup_idx
        ]

        self.val_qry_x = val_x[
            val_qry_idx
        ]

        self.val_qry_y = val_y[
            val_qry_idx
        ]

    # ======================================================
    # Support / Query split
    # ======================================================

    def _split_support_query(
        self,
        labels_tensor,
        rng
    ):

        labels_np = labels_tensor.cpu().numpy()

        indices = np.arange(
            len(labels_np)
        )

        support_indices = []
        query_indices = []

        if self.stratified:

            for label in np.unique(
                labels_np
            ):

                label_idx = indices[
                    labels_np == label
                ]

                rng.shuffle(
                    label_idx
                )

                split = int(
                    len(label_idx)
                    * self.support_ratio
                )

                # Ensure at least one support
                # sample when possible.
                if len(label_idx) > 1:
                    split = max(
                        1,
                        split
                    )

                support_indices.extend(
                    label_idx[:split]
                )

                query_indices.extend(
                    label_idx[split:]
                )

        else:

            rng.shuffle(
                indices
            )

            split = int(
                len(indices)
                * self.support_ratio
            )

            if len(indices) > 1:
                split = max(
                    1,
                    split
                )

            support_indices = indices[
                :split
            ]

            query_indices = indices[
                split:
            ]

        return (
            np.array(support_indices),
            np.array(query_indices)
        )

    # ======================================================
    # Dataset API
    # ======================================================

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(self, idx):

        x = self.data[idx]

        label = self.labels[idx]

        # No RP/image augmentation.
        #
        # x remains:
        # [1024, 6]

        return x, label

    # ======================================================
    # Meta-compatible dataloader
    # ======================================================

    def dataloader(
        self,
        validation=False
    ):

        if validation:

            sup_x = self.val_sup_x
            sup_y = self.val_sup_y

            qry_x = self.val_qry_x
            qry_y = self.val_qry_y

        else:

            sup_x = self.train_sup_x
            sup_y = self.train_sup_y

            qry_x = self.train_qry_x
            qry_y = self.train_qry_y

        # ==================================================
        # Support dataset
        # ==================================================

        support_dataset = copy.deepcopy(
            self
        )

        support_dataset.data = sup_x

        support_dataset.labels = sup_y

        support_dataset.train_mode = not validation

        # ==================================================
        # Query dataset
        # ==================================================

        query_dataset = copy.deepcopy(
            self
        )

        query_dataset.data = qry_x

        query_dataset.labels = qry_y

        query_dataset.train_mode = False

        # ==================================================
        # Support loader
        # ==================================================

        support_loader = DataLoader(
            support_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        # ==================================================
        # Query loader
        # ==================================================

        query_loader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return [
            support_loader,
            query_loader
        ]


# ==========================================================
# Optional non-LOSO dataset
# ==========================================================

class TimeSeriesDataset(Dataset):

    def __init__(
        self,
        dataset_file,
        train_ratio=0.8,
        support_ratio=0.3,
        stratified=True,
        seed=42,
        **kwargs,
    ):

        super(TimeSeriesDataset, self).__init__(
            name="TimeSeriesDataset",
            **kwargs
        )

        # ==================================================
        # Load data
        # ==================================================

        data = dataset_file["data"].float()

        labels = dataset_file["labels"].long()

        # ==================================================
        # Ensure [N, 1024, 6]
        # ==================================================

        if data.ndim != 3:

            raise ValueError(
                f"Expected data with 3 dimensions, "
                f"got {data.shape}"
            )

        if data.shape[-1] == 6:
            pass

        elif data.shape[1] == 6:

            data = data.transpose(
                1,
                2
            )

        else:

            raise ValueError(
                f"Expected data shape "
                f"[N, 1024, 6] or [N, 6, 1024], "
                f"got {data.shape}"
            )

        # ==================================================
        # Dataloader parameters
        # ==================================================

        self.batch_size = kwargs.get(
            "batch_size",
            32
        )

        self.shuffle = kwargs.get(
            "shuffle",
            True
        )

        self.num_workers = kwargs.get(
            "num_workers",
            1
        )

        # ==================================================
        # Parameters
        # ==================================================

        self.train_ratio = train_ratio

        self.support_ratio = support_ratio

        self.stratified = stratified

        self.seed = seed

        self.train_mode = True

        # ==================================================
        # Random generator
        # ==================================================

        rng = np.random.RandomState(
            seed
        )

        # ==================================================
        # Train / Validation split
        # ==================================================

        train_idx, val_idx = (
            self._split_train_val(
                labels,
                rng
            )
        )

        train_x = data[
            train_idx
        ]

        train_y = labels[
            train_idx
        ]

        val_x = data[
            val_idx
        ]

        val_y = labels[
            val_idx
        ]

        # ==================================================
        # Train -> Support / Query
        # ==================================================

        train_sup_idx, train_qry_idx = (
            self._split_support_query(
                train_y,
                rng
            )
        )

        self.train_sup_x = train_x[
            train_sup_idx
        ]

        self.train_sup_y = train_y[
            train_sup_idx
        ]

        self.train_qry_x = train_x[
            train_qry_idx
        ]

        self.train_qry_y = train_y[
            train_qry_idx
        ]

        # ==================================================
        # Validation -> Support / Query
        # ==================================================

        val_sup_idx, val_qry_idx = (
            self._split_support_query(
                val_y,
                rng
            )
        )

        self.val_sup_x = val_x[
            val_sup_idx
        ]

        self.val_sup_y = val_y[
            val_sup_idx
        ]

        self.val_qry_x = val_x[
            val_qry_idx
        ]

        self.val_qry_y = val_y[
            val_qry_idx
        ]

    # ======================================================
    # Train / Validation split
    # ======================================================

    def _split_train_val(
        self,
        labels,
        rng
    ):

        labels_np = labels.cpu().numpy()

        indices = np.arange(
            len(labels_np)
        )

        if self.stratified:

            train_idx = []
            val_idx = []

            for label in np.unique(
                labels_np
            ):

                label_indices = indices[
                    labels_np == label
                ]

                rng.shuffle(
                    label_indices
                )

                split = int(
                    len(label_indices)
                    * self.train_ratio
                )

                train_idx.extend(
                    label_indices[:split]
                )

                val_idx.extend(
                    label_indices[split:]
                )

        else:

            rng.shuffle(
                indices
            )

            split = int(
                len(indices)
                * self.train_ratio
            )

            train_idx = indices[
                :split
            ]

            val_idx = indices[
                split:
            ]

        return (
            np.array(train_idx),
            np.array(val_idx)
        )

    # ======================================================
    # Support / Query split
    # ======================================================

    def _split_support_query(
        self,
        labels,
        rng
    ):

        labels_np = labels.cpu().numpy()

        indices = np.arange(
            len(labels_np)
        )

        support_indices = []
        query_indices = []

        if self.stratified:

            for label in np.unique(
                labels_np
            ):

                label_idx = indices[
                    labels_np == label
                ]

                rng.shuffle(
                    label_idx
                )

                split = int(
                    len(label_idx)
                    * self.support_ratio
                )

                if len(label_idx) > 1:
                    split = max(
                        1,
                        split
                    )

                support_indices.extend(
                    label_idx[:split]
                )

                query_indices.extend(
                    label_idx[split:]
                )

        else:

            rng.shuffle(
                indices
            )

            split = int(
                len(indices)
                * self.support_ratio
            )

            if len(indices) > 1:
                split = max(
                    1,
                    split
                )

            support_indices = indices[
                :split
            ]

            query_indices = indices[
                split:
            ]

        return (
            np.array(support_indices),
            np.array(query_indices)
        )

    # ======================================================
    # Dataset API
    # ======================================================

    def __len__(self):

        return len(
            self.labels
        )

    def __getitem__(self, idx):

        x = self.data[idx]

        label = self.labels[idx]

        return x, label

    # ======================================================
    # Dataloader
    # ======================================================

    def dataloader(
        self,
        validation=False
    ):

        if validation:

            sup_x = self.val_sup_x
            sup_y = self.val_sup_y

            qry_x = self.val_qry_x
            qry_y = self.val_qry_y

        else:

            sup_x = self.train_sup_x
            sup_y = self.train_sup_y

            qry_x = self.train_qry_x
            qry_y = self.train_qry_y

        # ==================================================
        # Support
        # ==================================================

        support_dataset = copy.deepcopy(
            self
        )

        support_dataset.data = sup_x

        support_dataset.labels = sup_y

        # ==================================================
        # Query
        # ==================================================

        query_dataset = copy.deepcopy(
            self
        )

        query_dataset.data = qry_x

        query_dataset.labels = qry_y

        query_dataset.train_mode = False

        # ==================================================
        # Loaders
        # ==================================================

        support_loader = DataLoader(
            support_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

        query_loader = DataLoader(
            query_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return [
            support_loader,
            query_loader
        ]
from flautim.pytorch.Dataset import Dataset
import torch
import copy
import torchvision.transforms as T
from torch.utils.data import DataLoader
import flautim as fl
import random


class RecurrenceDataset(Dataset):
    def __init__(
        self,
        train_file,
        test_file,
        augment_multiplier=0,
        use_crop=False,
        use_erase=False,
        use_jitter=False,
        use_rotation=False,
        **kwargs,
    ):
        super(RecurrenceDataset, self).__init__(name="RecurrenceDataset", **kwargs)
        # Train set
        self.data = train_file["data"].float().permute(0, 3, 1, 2)
        self.labels = train_file["labels"].long()
        self.train_data = self.data
        self.train_labels = self.labels

        self.test_data = test_file["data"].float().permute(0, 3, 1, 2)
        self.test_labels = test_file["labels"].long()

        self.xdtype = torch.float32
        self.ydtype = torch.int64
        self.batch_size = kwargs.get("batch_size", 32)
        self.shuffle = kwargs.get("shuffle", True)
        self.num_workers = kwargs.get("num_workers", 1)

        # Augmentation options
        self.augment_multiplier = augment_multiplier
        self.use_crop = use_crop
        self.use_erase = use_erase
        self.use_jitter = use_jitter
        self.use_rotation = use_rotation

        self.angles = [90, 180, 270]
        self.jitter = T.ColorJitter(
            brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
        )
        self.random_crop = T.RandomResizedCrop(size=128, scale=(0.75, 0.95))
        self.erase = T.RandomErasing(
            p=1, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0
        )

    def _augment_image(self, img_tensor):
        """Apply a single augmentation pass to a (C,H,W) tensor."""
        if self.use_rotation:
            angle = random.choice(self.angles)
            k = angle // 90
            img_tensor = torch.rot90(img_tensor, k=k, dims=(1, 2))
        if self.use_crop:
            img_tensor = self.random_crop(img_tensor)
        if self.use_jitter:
            img_tensor[0:3] = self.jitter(img_tensor[0:3])
            img_tensor[3:6] = self.jitter(img_tensor[3:6])
        if self.use_erase:
            img_tensor = self.erase(img_tensor)
        return img_tensor

    def dataloader(self, validation=False):
        if not validation:
            train_set = self.train()

            # Apply augmentation only on training set
            if self.augment_multiplier > 0:
                augmented_data = []
                augmented_labels = []

                for i in range(len(train_set.data)):
                    img = train_set.data[i]
                    label = train_set.labels[i]

                    # keep original
                    augmented_data.append(img.clone())
                    augmented_labels.append(label.clone())

                    # add augmented versions
                    for _ in range(self.augment_multiplier):
                        aug_img = self._augment_image(img.clone())
                        augmented_data.append(aug_img)
                        augmented_labels.append(label.clone())

                train_set.data = torch.stack(augmented_data)
                train_set.labels = torch.stack(augmented_labels)

            return [
                DataLoader(
                    train_set,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                ),
                None,
            ]

        else:
            train_set = self.train()
            test_set = self.validation()

            train_loader = DataLoader(
                train_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
            )
            test_loader = DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            return [train_loader, test_loader]

    def train(self) -> Dataset:
        train = copy.deepcopy(self)
        train.data = self.train_data
        train.labels = self.train_labels
        return copy.deepcopy(train)

    def validation(self) -> Dataset:
        val = copy.deepcopy(self)
        val.data = self.test_data
        val.labels = self.test_labels
        return copy.deepcopy(val)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

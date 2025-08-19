import random
import copy

import flautim as fl
from flautim.pytorch.Dataset import Dataset
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader


class RecurrenceDataset(Dataset):
    def __init__(self,
                 train_data,
                 test_data,
                 augment_multiplier=0,
                 use_crop=False,
                 use_erase=False,
                 use_jitter=False,
                 use_rotation=False,
                 **kwargs):
        try:
            super(RecurrenceDataset, self).__init__(name="RecurrenceDataset", **kwargs)

            self.train_data = self.data = (
                train_data["data"].float().permute(0, 3, 1, 2)
            )  # Shape: (N, C, H, W)
            self.train_labels = self.labels = train_data["labels"].long()

            self.test_data = (
                test_data["data"].float().permute(0, 3, 1, 2)
            )  # Shape: (N, C, H, W)
            self.test_labels = test_data["labels"].long()

            self.xdtype = torch.float32
            self.ydtype = torch.int64
            self.shuffle = True
            self.batch_size = kwargs.get("batch_size", 16)
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
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset __init__: {e}")

    def _augment_image(self, img_tensor):
        """Apply a single augmentation pass to a (C,H,W) tensor."""
        try:
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
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset _augment_image: {e}")

    def dataloader(self, validation=False):
        if validation:
            dataset = self.validation()
        else:
            # Start with original training data
            dataset = self.train()

            if self.augment_multiplier > 0:
                augmented_data = []
                augmented_labels = []

                for i in range(len(dataset.data)):
                    img = dataset.data[i]
                    label = dataset.labels[i]

                    # always keep original
                    augmented_data.append(img.clone())
                    augmented_labels.append(label.clone())

                    # add augmented versions
                    for _ in range(self.augment_multiplier):
                        aug_img = self._augment_image(img.clone())
                        augmented_data.append(aug_img)
                        augmented_labels.append(label.clone())

                dataset.data = torch.stack(augmented_data)
                dataset.labels = torch.stack(augmented_labels)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=not validation,
            num_workers=self.num_workers,
        )

    def train(self):
        try:
            train = copy.deepcopy(self)
            train.data = self.train_data
            train.labels = self.train_labels
            return copy.deepcopy(train)
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset train(): {e}")

    def validation(self):
        try:
            val = copy.deepcopy(self)
            val.data = self.test_data
            val.labels = self.test_labels
            return copy.deepcopy(val)
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset validation(): {e}")

    def __len__(self):
        try:
            return len(self.labels)
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset __len__(): {e}")

    def __getitem__(self, idx):
        try:
            return self.data[idx], self.labels[idx]
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset __getitem__(): {e}")

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
        file,
        augment_multiplier=0,
        use_crop=False,
        use_erase=False,
        use_jitter=False,
        use_rotation=False,
        test_percentage=0,
        **kwargs,
    ):
        try:
            super(RecurrenceDataset, self).__init__(name="RecurrenceDataset", **kwargs)

            self.data = file["data"].float().permute(0, 3, 1, 2)  # Shape: (N, C, H, W)
            self.labels = file["labels"].long()

            self.xdtype = torch.float32
            self.ydtype = torch.int64
            self.batch_size = kwargs.get("batch_size", 32)
            self.shuffle = kwargs.get("shuffle", True)
            self.num_workers = kwargs.get("num_workers", 1)
            self.test_size = int(test_percentage * len(self.labels))

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
        try:
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
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset dataloader(): {e}")

    def train(self):
        try:
            train = copy.deepcopy(self)
            if self.test_size > 0:
                train.data = self.data[: -self.test_size]
                train.labels = self.labels[: -self.test_size]
            else:
                train.data = self.data
                train.labels = self.labels
            return train
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset train(): {e}")

    def validation(self):
        try:
            val = copy.deepcopy(self)
            if self.test_size > 0:
                val.data = self.data[-self.test_size :]
                val.labels = self.labels[-self.test_size :]
            else:
                val.data = []
                val.labels = []
            return val
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset validation(): {e}")

    def __len__(self):
        try:
            return len(self.labels)
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset __len__(): {e}")

    def __getitem__(self, idx):
        try:
            return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset __getitem__(): {e}")

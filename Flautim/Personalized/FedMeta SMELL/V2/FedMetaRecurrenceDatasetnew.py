from flautim.pytorch.Dataset import Dataset
import torch
import copy
import torchvision.transforms as T
from torch.utils.data import DataLoader
import flautim as fl
import random


class RecurrenceDatasetnew(Dataset):
    def __init__(
            self,
            train_file,
            test_file=None,
            use_crop=False,
            use_erase=False,
            use_jitter=False,
            use_rotation=False,
            **kwargs):
        try:
            super(RecurrenceDatasetnew, self).__init__(name="RecurrenceDatasetnew", **kwargs)

            # Train set
            self.data = train_file['data'].float().permute(0, 3, 1, 2)
            self.labels = train_file['labels'].long()

            # Optional test set
            if test_file is not None:
                self.test_data = test_file['data'].float().permute(0, 3, 1, 2)
                self.test_labels = test_file['labels'].long()
            else:
                self.test_data = None
                self.test_labels = None

            self.xdtype = torch.float32
            self.ydtype = torch.int64
            self.batch_size = kwargs.get('batch_size', 32)
            self.shuffle = kwargs.get('shuffle', True)
            self.num_workers = kwargs.get('num_workers', 1)

            # Augmentation options
            self.use_crop = use_crop
            self.use_erase = use_erase
            self.use_jitter = use_jitter
            self.use_rotation = use_rotation
            self.train_mode =True

            self.angles = [0, 90, 180, 270]
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
                # Apply jitter separately to both 3-channel halves
                img_tensor[:3] = self.jitter(img_tensor[:3])
                img_tensor[3:6] = self.jitter(img_tensor[3:6])
            if self.use_erase:
                img_tensor = self.erase(img_tensor)
            return img_tensor
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset _augment_image: {e}")

    def dataloader(self, validation=False):
        """Return [train_loader, test_loader] with on-the-fly augmentations."""
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        # On-the-fly augmentations only in training mode
        if self.train_mode and any([self.use_crop, self.use_erase, self.use_jitter, self.use_rotation]):
            img = self._augment_image(img.clone())

        return img, label

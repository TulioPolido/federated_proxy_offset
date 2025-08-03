from flautim.pytorch.Dataset import Dataset
import torch
import copy
import flautim as fl


class RecurrenceDataset(Dataset):
    def __init__(self, train_data, test_data, **kwargs):
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
        except Exception as e:
            fl.log(f"Error in RecurrenceDataset __init__: {e}")

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

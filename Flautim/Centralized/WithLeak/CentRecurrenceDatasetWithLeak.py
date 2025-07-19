from flautim.pytorch.Dataset import Dataset
import torch
import copy


class RecurrenceDataset(Dataset):
    def __init__(self, data, **kwargs):
        super(RecurrenceDataset, self)\
            .__init__(name="RecurrenceDataset", **kwargs)

        self.data = data['data'].float()\
            .permute(0, 3, 1, 2)  # Shape: (N, C, H, W)
        self.labels = data['labels'].long()

        self.xdtype = torch.float32
        self.ydtype = torch.int64
        self.shuffle = True
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_workers = kwargs.get('num_workers', 1)

        self.test_size = int(0.2 * len(self.labels))

    def train(self) -> Dataset:
        self.data = self.data[:-self.test_size]
        self.labels = self.labels[:-self.test_size]
        return copy.deepcopy(self)

    def validation(self) -> Dataset:
        self.data = self.data[-self.test_size:]
        self.labels = self.labels[-self.test_size:]
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

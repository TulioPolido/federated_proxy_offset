from flautim.pytorch.Dataset import Dataset
import torch
import copy


class RecurrenceDataset(Dataset):
    def __init__(self, file, **kwargs):
        super(RecurrenceDataset, self)\
            .__init__(name="RecurrenceDataset", **kwargs)

        self.data = file['data'].float()\
            .permute(0, 3, 1, 2)  # Shape: (N, C, H, W)
        self.labels = file['labels'].long()

        self.xdtype = torch.float32
        self.ydtype = torch.int64
        self.batch_size = kwargs.get('batch_size', 16)
        self.shuffle = kwargs.get('shuffle', True)
        self.num_workers = kwargs.get('num_workers', 1)

        self.test_size = int(0.2 * len(self.data))

    def train(self):
        train = copy.deepcopy(self)
        train.data = self.data[:-self.test_size]
        train.labels = self.labels[:-self.test_size]
        return copy.deepcopy(train)

    def validation(self):
        val = copy.deepcopy(self)
        val.data = self.data[-self.test_size:]
        val.labels = self.labels[-self.test_size:]
        return copy.deepcopy(val)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]), torch.tensor(self.labels[idx])

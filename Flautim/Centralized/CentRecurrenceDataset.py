from flautim.pytorch.Dataset import Dataset
import torch
import copy


class RecurrenceDataset(Dataset):
    def __init__(self, train_data, test_data, **kwargs):
        super(RecurrenceDataset, self)\
            .__init__(name="RecurrenceDataset", **kwargs)

        self.train_data = self.data = train_data['data'].float()\
            .permute(0, 3, 1, 2)  # Shape: (N, C, H, W)
        self.train_labels = self.labels = train_data['labels'].long()

        self.test_data = test_data['data'].float()\
            .permute(0, 3, 1, 2)  # Shape: (N, C, H, W)
        self.test_labels = test_data['labels'].long()

        self.xdtype = torch.float32
        self.ydtype = torch.int64
        self.shuffle = True
        self.batch_size = kwargs.get('batch_size', 16)
        self.num_workers = kwargs.get('num_workers', 1)

    def train(self):
        train = copy.deepcopy(self)
        train.data = self.train_data
        train.labels = self.train_labels
        return copy.deepcopy(train)

    def validation(self):
        val = copy.deepcopy(self)
        val.data = self.test_data
        val.labels = self.test_labels
        return copy.deepcopy(val)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

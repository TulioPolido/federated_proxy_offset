"""Handle basic dataset creation.

In case of PyTorch it should return dataloaders for your dataset (for both the clients
and the server). If you are using a custom dataset class, this module is the place to
define it. If your dataset requires to be downloaded (and this is not done
automatically -- e.g. as it is the case for many dataset in TorchVision) and
partitioned, please include all those functions and logic in the
`dataset_preparation.py` module. You can use all those functions from functions/methods
defined here of course.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from dataset_preparation import (
    _partition_data,
    split_train_validation_test_clients,
)

class FemnistDataset(Dataset):
    """
    [LEAF: A Benchmark for Federated Settings](https://github.com/TalwalkarLab/leaf).

    We imported the preprocessing method for the Femnist dataset from GitHub.
    """

    def __init__(self, dataset, transform):
        self.x = dataset["x"]
        self.y = dataset["y"]
        self.transform = transform

    def __getitem__(self, index):
        """Retrieve the input data and its corresponding label at a given index.

        Args:
            index (int): The index of the data item to fetch.

        Returns
        -------
            tuple:
                - input_data (torch.Tensor): Reshaped and optionally transformed data.
                - target_data (int or torch.Tensor): Label for the input data.
        """
        input_data = np.array(self.x[index]).reshape(28, 28)
        if self.transform:
            input_data = self.transform(input_data)
        target_data = self.y[index]
        return input_data.to(torch.float32), target_data

    def __len__(self):
        """Return the number of labels present in the dataset.

        Returns
        -------
            int: The total number of labels.
        """
        return len(self.y)


def load_datasets(
    path: str
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create the dataloaders to be fed into the model.

    Parameters
    ----------
    config: DictConfig
        data: float
            Used data type
        batch_size : int
            The size of the batches to be fed into the model,
            by default 10
        support_ratio : float
            The ratio of Support set for each client.(between 0 and 1)
            by default 0.2
    path : str
        The path where the leaf dataset was downloaded

    Returns
    -------
    Tuple[DataLoader, DataLoader, DataLoader]
    """
    data_type = "femnist"
    sup_ratio = 0.2
    batch_size = 32

    dataset = _partition_data(
        data_type=data_type, dir_path=path, support_ratio=sup_ratio
    )

    # Client list : 0.8, 0.1, 0.1
    clients_list = split_train_validation_test_clients(dataset[0]["users"])

    trainloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}
    valloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}
    testloaders: Dict[str, List[DataLoader]] = {"sup": [], "qry": []}

    if data_type == data_type:
        transform = transforms.Compose([transforms.ToTensor()])
        for user in clients_list[0]:
            trainloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=batch_size,
                    shuffle=True,
                )
            )
            trainloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
        for user in clients_list[1]:
            valloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
            valloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
        for user in clients_list[2]:
            testloaders["sup"].append(
                DataLoader(
                    FemnistDataset(dataset[0]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )
            testloaders["qry"].append(
                DataLoader(
                    FemnistDataset(dataset[1]["user_data"][user], transform),
                    batch_size=batch_size,
                )
            )

    return trainloaders, valloaders, testloaders

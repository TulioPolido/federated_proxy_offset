from flautim.pytorch.Model import Model
import torch.nn as nn
import torch

class FemnistNetwork(Model):
    """Convolutional Neural Network architecture.

    As described in Fei Chen 2018 paper :

    [FedMeta: Federated Meta-Learning with Fast Convergence and Efficient Communication]
    (https://arxiv.org/abs/1802.07876)
    """

    def __init__(self, context, **kwargs):
        super(FemnistNetwork, self)\
            .__init__(context, name="Femnist", **kwargs)
        
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=2
        )
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2))
        self.linear1 = nn.Linear(7 * 7 * 64, 2048)
        self.linear2 = nn.Linear(2048, 62)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the CNN.

        Parameters
        ----------
        x : torch.Tensor
            Input Tensor that will pass through the network

        Returns
        -------
        torch.Tensor
            The resulting Tensor after it has passed through the network
        """
        x = torch.relu(self.conv1(x))
        x = self.maxpool1(x)
        x = torch.relu(self.conv2(x))
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.relu((self.linear1(x)))
        x = self.linear2(x)
        return x
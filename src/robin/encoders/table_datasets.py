import torch
from torch import Tensor
from torch.utils.data import Dataset


class XDataset(Dataset):
    def __init__(self, data: Tensor, weights: Tensor):
        self.data = data
        self.weights = weights

    def __repr__(self):
        return f"Dataset: {self.data.shape}"

    def __getitem__(self, index):
        return self.data[index], self.weights[index]

    def __len__(self):
        return len(self.data)


class YXDataset(Dataset):
    def __init__(self, y: Dataset, x: Dataset):
        """Dataset for the input features and target labels.

        Args:
            y (Dataset): The target labels.
            x (Dataset): The input features.

        Raises:
            TypeError: If y or x is not a Dataset.
            ValueError: If y and x are not of the same length.
        """
        if not isinstance(y, Dataset) or not isinstance(x, Dataset):
            raise TypeError("y and x must be instances of Dataset")
        if not len(y) == len(x):
            raise ValueError("y and x must be of the same length")
        self.y = y
        self.x = x

    def __repr__(self):
        return f"{super().__repr__()}: {self.x.data.shape}, {self.y.data.shape}"

    def __getitem__(self, index):
        return self.y[index], self.x[index]

    def __len__(self):
        return len(self.y)


class ZDataset(Dataset):
    def __init__(self, num_samples: int, latent_size: int):
        self.z = torch.randn(num_samples, latent_size)

    def __repr__(self):
        return f"{super().__repr__()}: {self.z.shape}"

    def __getitem__(self, index):
        return self.z[index]

    def __len__(self):
        return len(self.z)


class YZDataset(Dataset):
    def __init__(self, y: Dataset, z: Dataset):
        """Dataset for the latent variables and target labels.

        Args:
            z (Dataset): The latent variables.
            y (Dataset): The target labels.

        Raises:
            TypeError: If y or z is not a Dataset.
            ValueError: If y and z are not of the same length.
        """
        if not isinstance(y, Dataset) or not isinstance(z, Dataset):
            raise TypeError("y and z must be instances of Dataset")
        if not len(y) == len(z):
            raise ValueError("y and z must be of the same length")
        self.y = y
        self.z = z

    def __repr__(self):
        return f"{super().__repr__()}: {self.z.data.shape}, {self.y.data.shape}"

    def __getitem__(self, index):
        return self.y[index], self.z[index]

    def __len__(self):
        return len(self.y)

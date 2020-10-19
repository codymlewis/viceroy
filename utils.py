"""
Utility functions for use on other classes in this project

Author: Cody
"""

import torch
from torch import nn
import torchvision


def load_data(ds_name, train=True):
    """
    Load the specified dataset in a form suitable for the Softmax model

    Keyword arguments:
    ds_name -- name of the dataset
    train -- load the training dataset if true otherwise load the validation
    """
    datasets = {
        "mnist": torchvision.datasets.MNIST,
    }
    if (chosen_set := datasets.get(ds_name)) is None:
        return torch.tensor(), torch.tensor()
    data = chosen_set(f"./data/{ds_name}", train=train, download=True)
    X = data.data
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    # return X.float(), nn.functional.one_hot(data.targets)
    return X.float(), data.targets.long().unsqueeze(dim=0)


def get_dims(x_shape, y_shape):
    """Get the dimensions for a dataset based on its shapes"""
    return {
        "x": x_shape[-1] if len(x_shape) > 1 else 1,
        "y": y_shape[-1] if len(y_shape) > 1 else 1,
    }


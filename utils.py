"""
Utility functions for use on other classes in this project

Author: Cody Lewis
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
    Y = data.targets.long().unsqueeze(dim=0)
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
    return {
        "x": X.float(),
        "y": Y,
        "x_dim": X.shape[-1] if len(X.shape) > 1 else 1,
        "y_dim": int(torch.max(Y)) + 1,
    }

# def find_acc(model, X, Y):
# torch.argmax(server.net.predict(val_data['x']), dim=1) == val_data['y'][0]
# count instances of true in above divide by length


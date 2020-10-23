"""
Utility functions for use on other classes in this project

Author: Cody Lewis
"""

from typing import NamedTuple
import json

import torch
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


def find_stats(model, X, Y, options):
    """Find statistics on the model based on validation data"""
    predictions = torch.argmax(model.predict(X), dim=1)
    accuracy = (predictions == Y[0]).sum().item() / len(Y[0])
    ids = Y[0] == options.adversaries['from']
    attack_success = (
        predictions[ids] == options.adversaries['to']
    ).sum().item() / len(ids)
    return {
        "accuracy": accuracy,
        "attack_success": attack_success
    }


def flatten_grads(grads):
    """Flatten gradients into vectors"""
    flat_grads = []
    for g in grads.values():
        t = torch.tensor([])
        for p in g['params']:
            t = torch.cat((t, p.flatten()))
        flat_grads.append(t)
    return flat_grads


def create_log(log_fn, stats):
    """Create the log file"""
    with open(log_fn, "w") as f:
        header = ""
        for k in stats.keys():
            header += k + ","
        f.write(header[:-1] + "\n")


def log_stats(log_fn, stats):
    """Log the statistics into the file"""
    with open(log_fn, "a") as f:
        f.write(str(list(stats.values()))[1:-1].replace(' ', '') + "\n")


class Options(NamedTuple):
    """Structure out the data from the options file"""
    server_epochs: int
    user_epochs: int
    users: int
    batch_size: int
    learning_rate: float
    fit_fun: str
    params: dict
    adversaries: dict
    verbosity: int
    result_log_file: str


def load_options():
    """Load a structure containing the options"""
    with open("options.json", "r") as f:
        options = json.loads(f.read())
        return Options(
            options['server_epochs'],
            options['user_epochs'],
            options['users'],
            options['batch_size'],
            options['learning_rate'],
            options['fit_fun'],
            options['params'],
            options['adversaries'],
            options['verbosity'],
            options['result_log_file']
        )
    return None

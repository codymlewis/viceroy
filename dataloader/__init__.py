import os
import logging

import numpy as np

from . import cifar10
from . import kddcup99
from . import mnist


def load(dataset, dir="data"):
    """
    Load a dataset in the form (X, y, train) where X is the collection of samples,
    y is the collection of labels, and train are indices within the collections that
    pertain to the training dataset. The dataset is downloaded if it does not yet exist
    at <dir>/<dataset>.
    Arguments:
    - dataset: the name of the dataset to load
    - dir: the directory where the dataset is/will be stored
    """
    fn = f"{dir}/{dataset}.npz"
    if not os.path.exists(fn):
        download(dir, dataset)
    ds = np.load(f"{dir}/{dataset}.npz")
    return ds['X'], ds['y'], ds['train']


def download(dir, dataset):
    """Download a dataset to a directory."""
    if globals().get(dataset) is None:
        logging.error('Dataset %s not found', dataset)
        raise ValueError(f"Dataset {dataset} not found")
    globals()[dataset].download(f"{dir}/{dataset}")

"""
The CIFAR-10 object recognition dataset from https://www.cs.toronto.edu/~kriz/cifar.html.
Contains small 32x32 RGB images of 10 classes.
"""

import os

import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np
import logging


def load():
    """Load the CIFAR-10 dataset."""
    X, y = skds.fetch_openml('CIFAR_10', return_X_y=True)
    return X.to_numpy(), y.to_numpy()


def preprocess(X, y):
    """Preprocess the CIFAR-10 dataset."""
    X = skp.MinMaxScaler().fit_transform(X)
    X = X.reshape(-1, 32, 32, 3).astype(np.float32)
    y = skp.LabelEncoder().fit_transform(y).astype(np.int8)
    return X, y


def download(path):
    """Download, preprocess, and save the dataset."""
    fn = os.path.expanduser(path)
    dir = os.path.dirname(fn)

    logging.info("Downloading data...")
    X, y = load()
    logging.info("Done. Preprocessing data...")
    X, y = preprocess(X, y)
    logging.info(f"Done. Saving as a compressed file to {fn}")
    os.makedirs(dir, exist_ok=True)
    np.savez_compressed(fn, X=X, y=y, train=(np.arange(len(y)) < 50_000))
    logging.info("Finished dataset download.")

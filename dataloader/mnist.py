"""
MNIST digit recognition dataset from http://yann.lecun.com/exdb/mnist/.
"""

import os

import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np
import logging


def load():
    """Load the MNIST dataset."""
    X, y = skds.fetch_openml('mnist_784', return_X_y=True)
    return X.to_numpy(), y.to_numpy()


def preprocess(X, y):
    """Preprocess the MNIST dataset."""
    X = skp.MinMaxScaler().fit_transform(X)
    X = X.reshape(-1, 28, 28, 1).astype(np.float32)
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
    np.savez_compressed(fn, X=X, y=y, train=(np.arange(len(y)) < 60_000))
    logging.info("Finished dataset download.")

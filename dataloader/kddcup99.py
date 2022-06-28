"""
KDD Cup '99 intrusion detection dataset http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html.
"""


import os

import sklearn.datasets as skds
import sklearn.preprocessing as skp
import numpy as np
import logging


def load():
    """Load the KDD Cup '99 dataset."""
    return skds.fetch_kddcup99(return_X_y=True)


def preprocess(X, y):
    """Preprocess the KDD Cup '99 dataset."""
    idx = (y != b'spy.') & (y != b'warezclient.')
    X, y = X[idx], y[idx]
    y = skp.LabelEncoder().fit_transform(y).astype(np.int8)
    for i in [1, 2, 3]:
        X[:, i] = skp.LabelEncoder().fit_transform(X[:, i])
    X = skp.MinMaxScaler().fit_transform(X)
    X = X.astype(np.float32)
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
    np.savez_compressed(fn, X=X, y=y, train=(np.arange(len(y)) < 345_815))
    logging.info("Finished dataset download.")

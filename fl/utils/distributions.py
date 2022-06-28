"""
Federated learning data distribution mapping functions.

All functions take the following arguments:
- X: the samples
- y: the labels
- nendpoints: the number of endpoints
- nclasses: the number of classes
- rng: the random number generator

And they all return a list of lists of indices, where the outer list is indexed by endpoint.
"""

import itertools

import numpy as np
import logging


def homogeneous(X, y, nendpoints, nclasses, rng):
    """Assign all data to all endpoints"""
    return [np.arange(len(y)) for _ in range(nendpoints)]


def extreme_heterogeneous(X, y, nendpoints, nclasses, rng):
    """Assign each endpoint only the data from each class"""
    return [np.isin(y, i % nclasses) for i in range(nendpoints)]


def lda(X, y, nendpoints, nclasses, rng, alpha=0.5):
    r"""
    Latent Dirichlet allocation defined in `https://arxiv.org/abs/1909.06335 <https://arxiv.org/abs/1909.06335>`_
    default value from `https://arxiv.org/abs/2002.06440 <https://arxiv.org/abs/2002.06440>`_

    Optional arguments:
    - alpha: the $\alpha$ parameter of the Dirichlet function,
    the distribution is more i.i.d. as $\alpha \to \infty$ and less i.i.d. as $\alpha \to 0$
    """
    distribution = [[] for _ in range(nendpoints)]
    proportions = rng.dirichlet(np.repeat(alpha, nendpoints), size=nclasses)
    for c in range(nclasses):
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        dists_c = np.split(idx_c, np.round(np.cumsum(proportions[c]) * len(idx_c)).astype(int)[:-1])
        distribution = [distribution[i] + d.tolist() for i, d in enumerate(dists_c)]
    logging.debug(f"distribution: {proportions}")
    return distribution


def iid_partition(X, y, nendpoints, nclasses, rng):
    """Assign each endpoint iid the data from each class as defined in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_"""
    idx = np.arange(len(y))
    rng.shuffle(idx)
    return np.split(idx, [round(i * (len(y) // nendpoints)) for i in range(1, nendpoints)])


def shard(X, y, nendpoints, nclasses, rng, shards_per_endpoint=2):
    """
    The shard data distribution scheme as defined in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_
    shards are even partitions of the data after sorting by class.

    Optional arguments:
    - shards_per_endpoint: the number of shards to assign to each endpoint.
    """
    idx = np.argsort(y)  # sort by label
    shards = np.split(idx, [round(i * (len(y) // (nendpoints * shards_per_endpoint))) for i in range(1, nendpoints * shards_per_endpoint)])
    assignment = rng.choice(np.arange(len(shards)), (nendpoints, shards_per_endpoint), replace=False)
    return [list(itertools.chain(*[shards[assignment[i][j]] for j in range(shards_per_endpoint)])) for i in range(nendpoints)]


def assign_classes(X, y, nendpoints, nclasses, rng, classes=None):
    """
    Assign each endpoint only the data from the list specified class

    Arguments:
    - classes: a list of classes to assign to each endpoint.
    """
    if classes is None:
        raise ValueError("Classes not specified in distribution")
    return [np.isin(y, classes[i]) for i in range(nendpoints)]
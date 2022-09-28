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

import numpy as np
import logging


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


def assign_classes(X, y, nendpoints, nclasses, rng, classes=None):
    """
    Assign each endpoint only the data from the list specified class

    Arguments:
    - classes: a list of classes to assign to each endpoint.
    """
    if classes is None:
        raise ValueError("Classes not specified in distribution")
    return [np.isin(y, classes[i]) for i in range(nendpoints)]

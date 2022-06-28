"""
The multi-Krum algorithm proposed in `https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html <https://papers.nips.cc/paper/2017/hash/f4b9ec30ad9f68f89b29639786cb62ef-Abstract.html>`_
it is designed to be robust to Byzantine faults with i.i.d. environments.
"""

import numpy as np
import jax

from . import captain


class Captain(captain.ScaleCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng(), clip=3):
        """
        Construct the Krum captain.

        Optional arguments:
        - clip: the number of expected faults in each round.
        """
        super().__init__(params, opt, opt_state, network, rng)
        self.clip = clip

    def update(self, all_grads):
        pass

    def scale(self, all_grads):
        n = len(all_grads)
        X = np.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
        scores = np.zeros(n)
        distances = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None] - 2 * np.dot(X, X.T)
        for i in range(len(X)):
            scores[i] = np.sum(np.sort(distances[i])[1:((n - self.clip) - 1)])
        idx = np.argpartition(scores, n - self.clip)[:(n - self.clip)]
        alpha = np.zeros(n)
        alpha[idx] = 1
        return alpha
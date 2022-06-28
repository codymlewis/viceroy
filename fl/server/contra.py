"""
The CONTRA algorithm proposed in `https://www.ittc.ku.edu/~bluo/pubs/Awan2021ESORICS.pdf <https://www.ittc.ku.edu/~bluo/pubs/Awan2021ESORICS.pdf>`_
it is designed to provide robustness to poisoning adversaries within many statistically heterogenous environments.
"""

import numpy as np
import sklearn.metrics.pairwise as smp
import jax
import jax.numpy as jnp

from . import captain


class Captain(captain.ScaleCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng(), C=0.1, k=10, delta=0.1, t=0.5):
        """
        Construct the CONTRA captain.

        Optional arguments:

        - C: Percentage of collaborators to be selected for each update.
        - k: Number of expected adversarial collaborators.
        - delta: Amount the increase/decrease the reputation (selection likelyhood) by.
        - t: Threshold for choosing when to increase the reputation.
        """
        super().__init__(params, opt, opt_state, network, rng)
        self.histories = jnp.zeros((len(network), jax.flatten_util.ravel_pytree(params)[0].shape[0]))
        self.C = C
        self.k = round(k * C)
        self.lamb = C * (1 - C)
        self.delta = delta
        self.t = t
        self.reps = np.ones(len(network))
        self.J = round(self.C * len(network))

    def update(self, all_grads):
        """Update the stored collaborator histories, that is, perform $H_{i, t + 1} \gets H_{i, t} + \Delta_{i, t + 1} : \\forall i \in \mathcal{U}$"""
        self.histories = update(self.histories, all_grads)

    def scale(self, all_grads):
        n_clients = self.histories.shape[0]
        p = self.C + self.lamb * self.reps
        p[p == 0] = 0
        p = p / p.sum()
        idx = np.random.choice(n_clients, size=self.J, p=p)
        L = idx.shape[0]
        cs = abs(smp.cosine_similarity(self.histories[idx])) - np.eye(L)
        cs[cs < 0] = 0
        taus = (-np.partition(-cs, self.k - 1, axis=1)[:, :self.k]).mean(axis=1)
        self.reps[idx] = np.where(taus > self.t, self.reps[idx] + self.delta, self.reps[idx] - self.delta)
        cs = cs * np.minimum(1, taus[:, None] / taus)
        taus = (-np.partition(-cs, self.k - 1, axis=1)[:, :self.k]).mean(axis=1)
        lr = np.zeros(n_clients)
        lr[idx] = 1 - taus
        self.reps[idx] = self.reps[idx] / self.reps[idx].max()
        lr[idx] = lr[idx] / lr[idx].max()
        lr[(lr == 1)] = .99  # eliminate division by zero in logit
        lr[idx] = np.log(lr[idx] / (1 - lr[idx])) + 0.5
        lr[(np.isinf(lr) + lr > 1)] = 1
        lr[(lr < 0)] = 0
        return lr


@jax.jit
def update(histories, all_grads):
    """Perform histories + all_grads, elementwise."""
    return jnp.array([h + jax.flatten_util.ravel_pytree(g)[0] for h, g in zip(histories, all_grads)])
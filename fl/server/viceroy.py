"""
Viceroy algorithm for mitigating the impact of the on-off toggling attack.
"""

import sys
from functools import partial

import numpy as np
import sklearn.metrics.pairwise as smp

import jax
import jax.numpy as jnp

from . import captain


class Captain(captain.ScaleCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng(), tau_0=56, tau_1=5):
        r"""
        Construct the Viceroy captain.

        Optional arguments:
        - tau_0: amount of rounds for the reputation to decay to 0 ($\tau_0$).
        - tau_1: amount of rounds for the reputation to build to 1 ($\tau_1$).
        """
        super().__init__(params, opt, opt_state, network, rng)
        self.histories = jnp.zeros((len(network), jax.flatten_util.ravel_pytree(params)[0].shape[0]))
        self.reps = np.array([1.0 for _ in range(len(network))])
        self.round = 1
        self.omega = (abs(sys.float_info.epsilon))**(1/tau_0)
        self.eta = 1 / tau_1

    def update(self, all_grads):
        if self.round > 1:
            history_scale = scale(self.histories)
            current_scale = scale(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads]))
            self.reps = jnp.clip(self.reps + ((1 - 2 * abs(history_scale - current_scale)) / 2) * self.eta, 0, 1)
        self.round += 1
        self.histories = update(self.omega, self.histories, all_grads)

    def scale(self, all_grads):
        return (self.reps * scale(self.histories)) + ((1 - self.reps) * scale(jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])))


def scale(X):
    """A modified FoolsGold algorithm for scaling the gradients/histories."""
    n_clients = X.shape[0]
    cs = smp.cosine_similarity(X) - np.eye(n_clients)
    maxcs = np.max(cs, axis=1)
    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0
    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99
    # Logit function
    idx = wv != 0
    wv[idx] = (np.log(wv[idx] / (1 - wv[idx])) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0
    return wv

@partial(jax.jit, static_argnums=(0,))
def update(omega, histories, all_grads):
    r"""A decayed histories update, $H_{i, t + 1} \gets \omega H_{i, t} + \Delta_{i, t + 1} : \forall i \in \mathcal{U}$."""
    return jnp.array([omega * h + jax.flatten_util.ravel_pytree(g)[0] for h, g in zip(histories, all_grads)])
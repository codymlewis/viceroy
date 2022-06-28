"""
Basic federated averaging proposed in `https://arxiv.org/abs/1602.05629 <https://arxiv.org/abs/1602.05629>`_
this simply scales received gradients by the number of data they trained on divided by the total number of data,
$\\frac{n_i}{\sum_{i \in \mathcal{U}} n_i}$.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import captain


class Captain(captain.ScaleCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng()):
        super().__init__(params, opt, opt_state, network, rng)
        self.batch_sizes = jnp.array([c.batch_size * c.epochs for c in network.clients])
    
    def update(self, all_grads):
        """Update the stored batch sizes ($n_i$)."""
        self.batch_sizes = jnp.array([c.batch_size * c.epochs for c in self.network.clients])

    def scale(self, all_grads):
        return jax.vmap(lambda b: b / self.batch_sizes.sum())(self.batch_sizes)
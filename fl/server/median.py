"""
Median aggregation method proposed in `https://arxiv.org/abs/1803.01498 <https://arxiv.org/abs/1803.01498>`_
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import captain


class Captain(captain.AggregateCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng()):
        super().__init__(params, opt, opt_state, network, rng)
        self.G_unraveller = jax.jit(jax.flatten_util.ravel_pytree(params)[1])

    def step(self):
        # Client side updates
        all_grads = self.network(self.params, self.rng, return_weights=False)

        # Server side updates
        gs = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
        med_grad = self.G_unraveller(median(gs))
        self.params, self.opt_state = self.update_params(self.params, self.opt_state, med_grad)
        return 0, all_grads

    def update(self, all_weights):
        pass

    def scale(self, *args):
        return 0


@jax.jit
def median(xs):
    """Take the elementwise median of a 2D array"""
    return jnp.median(xs, axis=0)

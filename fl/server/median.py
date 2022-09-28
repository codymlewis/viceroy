"""
Median aggregation method proposed in https://arxiv.org/abs/1803.01498
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import server


class Server(server.Server):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng()):
        super().__init__(params, opt, opt_state, network, rng)
        self.G_unraveller = jax.jit(jax.flatten_util.ravel_pytree(params)[1])

    def step(self):  # Due to being a bit different to the others we redefine the step method
        # Client side updates
        all_grads = self.network(self.params, self.rng, return_weights=False)

        # Server side updates
        gs = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
        med_g = median(gs)
        alpha = calc_alpha(med_g, gs)
        med_grad = self.G_unraveller(med_g)
        self.params, self.opt_state = self.update_params(self.params, self.opt_state, med_grad)
        return alpha, all_grads

    def update(self, all_grads):
        pass  # stateless

    def scale(self, all_grads):
        gs = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
        med_g = median(gs)
        return calc_alpha(med_g, gs)


@jax.jit
def median(xs):
    """Take the elementwise median of a 2D array"""
    return jnp.median(xs, axis=0)


@jax.jit
def calc_alpha(m, xs):
    return jnp.mean(m == xs, axis=1)

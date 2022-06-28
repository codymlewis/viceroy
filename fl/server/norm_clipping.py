"""
Static norm clipping aggregator proposed in `https://arxiv.org/abs/1911.07963 <https://arxiv.org/abs/1911.07963>`_,
scales down any updates that sit out side of the $l_2$ sphere of radius $M$.
"""

import jax
import jax.numpy as jnp
import numpy as np

from . import captain


class Captain(captain.ScaleCaptain):
    def __init__(self, params, opt, opt_state, network, rng=np.random.default_rng(), M=1.0):
        """
        Construct the norm clipping aggregator.

        Optional arguments:
        - M: the radius of the $l_2$ sphere to scale according to.
        """
        super().__init__(params, opt, opt_state, network, rng)
        self.M = M
    
    def update(self, all_grads):
        pass

    def scale(self, all_grads):
        G = jnp.array([jax.flatten_util.ravel_pytree(g)[0] for g in all_grads])
        return jax.vmap(lambda g: 1 / jnp.maximum(1, jnp.linalg.norm(g, ord=2) / self.M))(G)
"""
Federated learning free rider attack proposed in `https://arxiv.org/abs/1911.12560 <https://arxiv.org/abs/1911.12560>`_
"""


import fl.lib

import numpy as np


def convert(client, attack_type, params, rng=np.random.default_rng()):
    """
    Convert an endpoint into a free rider adversary.

    Arguments:
    - client: the endpoint to convert
    - attack_type: the attack type to use, options are "random", "delta, and "advanced delta"
    - params: the parameters of the starting model
    - rng: the random number generator to use
    """
    client.attack_type = attack_type
    client.prev_params = params
    client.rng = rng
    client.update = update(client.opt).__get__(client)


def update(opt):
    """Free rider update function for endpoints."""
    def _apply(self, params, opt_state, X, y):
        if self.attack_type == "random":
            grads = fl.lib.tree_uniform(params, low=-10e-3, high=10e-3, rng=self.rng)
        else:
            grads = fl.lib.tree_add(params, fl.lib.tree_mul(self.prev_params, -1))
            if "advanced" in self.attack_type:
                grads = fl.lib.tree_add_normal(grads, loc=0.0, scale=10e-4, rng=self.rng)
        updates, opt_state = opt.update(grads, opt_state, params)
        self.prev_params = params
        return grads, opt_state, updates
    return _apply

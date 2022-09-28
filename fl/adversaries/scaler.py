"""
Scale the updates submitted from selected endpoints.
"""


import numpy as np

from fl import server

import fl.utils.functions


def convert(client, num_endpoints):
    """A simple naive scaled model replacement attack."""
    client.quantum_update = client.update
    client.update = lambda p, o, X, y: update(client.opt, num_endpoints, p, o, client.quantum_update(p, o, X, y)[0])


def update(opt, scale, params, opt_state, grads):
    """Scale the gradient and resulting update."""
    grads = fl.utils.functions.tree_mul(grads, scale)
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates


class GradientTransform:
    """
    Gradient transform that scales updates based on the inverse of the result from the aggregation scale value.
    """
    def __init__(self, params, opt, opt_state, network, alg, num_adversaries, rng=np.random.default_rng(), **kwargs):
        """
        Construct the gradient transform.

        Arguments:
        - params: the parameters of the starting model
        - opt: the optimizer to use
        - opt_state: the optimizer state
        - network: the network of the FL environment
        - alg: the FL aggregation algorithm to use
        - num_adversaries: the number of adversaries
        - rng: the random number generator to use
        """
        self.num_adv = num_adversaries
        self.alg = alg
        self.server = getattr(server, self.alg).Server(params, opt, opt_state, network, rng, **kwargs)

    def __call__(self, all_grads):
        """Get the scale value and scale the gradients."""
        self.server.update(all_grads)
        alpha = np.array(self.server.scale(all_grads))
        idx = np.arange(len(alpha) - self.num_adv, len(alpha))[alpha[-self.num_adv:] > 0.0001]
        alpha[idx] = 1 / alpha[idx]
        for i in idx:
            all_grads[i] = fl.utils.functions.tree_mul(all_grads[i], alpha[i])
        return all_grads

"""
Alternating minimization model poisoning, proposed in `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_
"""

from functools import partial

import jax
import optax

import fl.client.scout
import fl.lib


def convert(client,  poison_epochs, stealth_epochs, stealth_data):
    """
    Convert an endpoint into an alternating minimization adversary.
    
    Arguments:
    - client: the endpoint to convert
    - poison_epochs: the number of epochs to run the poisoned training for
    - stealth_epochs: the number of epochs to run the stealth training for
    - stealth_data: a generator that yields the stealth data
    """
    client.poison_update = client.update
    client.stealth_update = partial(fl.client.scout.update, client.opt, client.loss)
    client.poison_epochs = poison_epochs
    client.stealth_epochs = stealth_epochs
    client.stealth_data = stealth_data
    client.update = update.__get__(client)


def update(self, params, opt_state, X, y):
    """Alternating minimization update function for endpoints."""
    sum_grads = None
    for _ in range(self.poison_epochs):
        grads, opt_state, updates = self.poison_update(params, opt_state, X, y)
        params = optax.apply_updates(params, updates)
        sum_grads = grads if sum_grads is None else fl.lib.tree_add(sum_grads, grads)
    for _ in range(self.stealth_epochs):
        grads, opt_state, updates = self.stealth_update(params, opt_state, *next(self.stealth_data))
        params = optax.apply_updates(params, updates)
        sum_grads = grads if sum_grads is None else fl.lib.tree_add(sum_grads, grads)
    return sum_grads, opt_state, updates

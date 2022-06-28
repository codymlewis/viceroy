"""
Targeted model poisoning (label flipping) attack, proposed in `https://arxiv.org/abs/1811.12470 <https://arxiv.org/abs/1811.12470>`_
"""

from functools import partial

import jax


def convert(client, dataset, attack_from, attack_to):
    """
    Convert an endpoint into a label flipping adversary.

    Arguments:
    - client: the endpoint to convert
    - dataset: the dataset to use
    - attack_from: the label to attack
    - attack_to: the label to replace the attack_from label with
    """
    data = dataset.get_iter(
        "train",
        client.batch_size,
        filter=lambda y: y == attack_from,
        map=partial(labelflip_map, attack_from, attack_to)
    )
    client.update = partial(update, client.opt, client.loss, data)


def labelflip_map(attack_from, attack_to, X, y):
    """Map function for converting a dataset to a label flipping dataset."""
    idfrom = y == attack_from
    y[idfrom] = attack_to
    return (X, y)


@partial(jax.jit, static_argnums=(0, 1, 2,))
def update(opt, loss, data, params, opt_state, X, y):
    """Label flipping update function for endpoints."""
    grads = jax.grad(loss)(params, *next(data))
    updates, opt_state = opt.update(grads, opt_state, params)
    return grads, opt_state, updates
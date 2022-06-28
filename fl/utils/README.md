# MP
Shared utilities to be used between both the endpoints and the server

## [Compression](mp/compression)
Functions and transforms for compressing the internal communications of federated learning.

## [Datasets](mp/datasets)
Contains functions and classes to handle the loading, division, and interface of datasets.

## [Distributions](mp/distributions)
Contains functions to handle the generation of distributions of data across endpoints.

## [Losses](mp/losses)
Collection of jax-based loss functions, they have a curried format so the network may be saved for a jit compiled function

## [Metrics](mp/metrics)
Collection of functions for measuring the performance of the federated learning, also includes an object to perform running measurements of the
system performance.

## [Network](mp/network)
Allows for the definition of the network structure for the FL process. Generally the following steps are performed; first the `Network` object
is constructed using an optimizer and loss function as arguments, next controllers are added, then clients are added to those controllers. Where
controllers are a non-training node that connects to other controllers and a collection of the clients, they orchestrate the passing of gradients.

The following snippet shows the construction of a star network,

```python
network = ymir.mp.network.Network(opt, loss)
network.add_controller("main", is_server=True)
for d in data:
    network.add_host("main", ymir.regiment.Scout(opt, opt_state, loss, d, epochs))
```

## [Optimizers](mp/optimizers)
Collection of optax-based optimizers to be used on both endpoints and the server
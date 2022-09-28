import jax
import haiku as hk


def LeNet_300_100(classes, x):
    """LeNet 300-100 network from `https://doi.org/10.1109/5.726791 <https://doi.org/10.1109/5.726791>`_"""
    x = hk.Sequential([
        hk.Flatten(),
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
    ])(x)
    return hk.Linear(classes)(x)


def ConvLeNet(classes, x):
    """LeNet 300-100 network with a convolutional layer and max pooling layer prepended"""
    x = hk.Sequential([
        hk.Conv2D(64, kernel_shape=11, stride=4), jax.nn.relu,
        hk.MaxPool(3, strides=2, padding="VALID"),
        hk.Flatten(),
        hk.Linear(300), jax.nn.relu,
        hk.Linear(100), jax.nn.relu,
    ])(x)
    return hk.Linear(classes)(x)

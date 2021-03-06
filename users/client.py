"""
Classes and functions for the client networking aspect of federated learning

Author: Cody Lewis
"""

from utils.models import load_model
from utils.datasets import load_data


class Client:
    """Federated learning client"""
    def __init__(self, options, classes):
        self.data = load_data(options, train=True, classes=classes)
        params = options.model_params
        self.net = load_model(params)
        self.options = options

    def fit(self, scaling=1, verbose=False):
        """Fit the client to its own copy of data"""
        return self.net.fit(
            self.data['dataloader'],
            self.options.user_epochs,
            scaling=scaling,
            verbose=verbose
        )

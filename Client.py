"""
Classes and functions for the client networking aspect of federated learning

Author: Cody Lewis
"""

import SoftMaxModel


class Client:
    def __init__(self, data):
        self.net = SoftMaxModel.SoftMaxModel(data['x_dim'], data['y_dim'])
        self.x = data['x']
        self.y = data['y']

    def fit(self, verbose=True):
        return self.net.fit(self.x, self.y, 8, 1, verbose=verbose)

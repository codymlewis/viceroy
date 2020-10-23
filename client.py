"""
Classes and functions for the client networking aspect of federated learning

Author: Cody Lewis
"""

from functools import reduce

import torch

from softmax_model import SoftMaxModel


class Client:
    """Federated learning client"""
    def __init__(self, data, options, classes):
        self.net = SoftMaxModel(
            data['x_dim'],
            data['y_dim'],
            lr=options.learning_rate
        )
        ids = reduce(
            (lambda x, y: x + (data['y'][0] == y)),
            [torch.zeros(len(data['y'][0])).bool()] + classes
        )
        self.x = data['x'][ids]
        self.y = data['y'][0][ids].unsqueeze(dim=0)
        self.options = options
        self.latest_loss = 0
        self.latest_grad = 0

    def fit(self, verbose=False):
        """Fit the client to its own copy of data"""
        return self.net.fit(
            self.x,
            self.y,
            self.options.batch_size,
            self.options.user_epochs,
            verbose=verbose
        )

    def fit_async(self, verbose=False):
        """Run the fit method in a suitable way for async running"""
        self.latest_loss, self.latest_grad = self.fit(verbose=verbose)


class Flipper(Client):
    """A simple label-flipping model poisoner"""
    def __init__(self, data, options, classes):
        super().__init__(data, options, classes)
        ids = data['y'][0] == options.adversaries['from']
        self.x = data['x'][ids]
        self.y = torch.tensor(
            [options.adversaries['to'] for _ in ids]
        ).unsqueeze(dim=0)


class OnOff(Client):
    """
    Label flipping poisoner that switches its attack on and off every few
    epochs
    """
    def __init__(self, data, options, classes):
        super().__init__(data, options, classes)
        ids = data['y'][0] == options.adversaries['from']
        self.shadow_x = data['x'][ids]
        self.shadow_y = torch.tensor(
            [options.adversaries['to'] for _ in ids]
        ).unsqueeze(dim=0)
        self.epochs = 0

    def fit(self, verbose=False):
        self.epochs += 1
        if self.epochs % self.options.adversaries['toggle_time'] == 0:
            temp = self.x
            self.x = self.shadow_x
            self.shadow_x = temp
            temp = self.y
            self.y = self.shadow_y
            self.shadow_y = temp
        return super().fit(verbose=verbose)


# Dictionary for factory stuctures of adversary construction
ADVERSARY_TYPES = {
    "flip": Flipper,
    "onoff": OnOff
}

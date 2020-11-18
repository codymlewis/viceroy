"""
Defines the adversaries within the system

Author: Cody Lewis
"""


import torch

from client import Client


class Flipper(Client):
    """A simple label-flipping model poisoner"""
    def __init__(self, options, classes):
        super().__init__(options, [options.adversaries['from']])
        self.data['dataloader'].dataset.targets[:] = options.adversaries['to']


class OnOff(Client):
    """
    Label flipping poisoner that switches its attack on and off every few
    epochs
    """
    def __init__(self, options, classes):
        super().__init__(options, classes)
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
    "label flip": Flipper,
    "on off": OnOff
}

"""
Defines the adversaries within the system

Author: Cody Lewis
"""


from itertools import cycle

from client import Client
from datasets import load_data
import errors


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
        self.shadow_data = load_data(options, [options.adversaries['from']])
        self.shadow_data['dataloader'].dataset.targets[:] = \
            options.adversaries['to']
        self.toggle_time = cycle(self.options.adversaries['toggle_times'])
        self.epochs = 0
        self.next_switch = self.epochs + next(self.toggle_time)

    def fit(self, verbose=False):
        if self.epochs == self.next_switch:
            temp = self.data
            self.data = self.shadow_data
            self.shadow_data = temp
            self.next_switch += next(self.toggle_time)
        self.epochs += 1
        return super().fit(verbose=verbose)


def load_adversary(adversary_name):
    adversaries = {
        "label flip": Flipper,
        "on off": OnOff,
    }
    if (chosen_adversary := adversaries.get(adversary_name)) is None:
        raise errors.MisconfigurationError(
            f"Model '{adversary_name}' does not exist, " +
            f"possible options: {set(adversaries.keys())}"
        )
    return chosen_adversary

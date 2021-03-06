"""
Defines the adversaries within the system and a function to load them

Author: Cody Lewis
"""


from itertools import cycle

from users.client import Client
from utils.datasets import load_data


class OnOff(Client):
    """
    Label flipping poisoner that switches its attack on and off every few
    epochs
    """
    def __init__(self, options, classes):
        super().__init__(options, classes)
        self.shadow_data = load_data(
            options,
            [options.adversaries['from']],
            backdoor=options.adversaries['type'].find('backdoor') >= 0
        )
        self.shadow_data['dataloader'].dataset.targets[:] = \
            options.adversaries['to']
        if (tt := self.options.adversaries['toggle_times']):
            self.toggle_time = cycle(tt)
        else:
            self.toggle_time = cycle([0])
        self.epochs = 0
        self.attacking = False
        if self.options.adversaries['delay'] is None:
            self.next_switch = self.epochs + next(self.toggle_time)
        else:
            self.next_switch = self.epochs + self.options.adversaries['delay']
            next(self.toggle_time)

    def fit(self, verbose=False):
        if self.epochs == self.next_switch:
            self.attacking = not self.attacking
            temp = self.data
            self.data = self.shadow_data
            self.shadow_data = temp
            self.next_switch += next(self.toggle_time)
        self.epochs += 1
        scale_up = 1
        if self.attacking and self.options.adversaries['scale_up']:
            batch_size = self.options.model_params['batch_size']
            scale_up = (batch_size * self.options.users) / batch_size
        return super().fit(scaling=scale_up, verbose=verbose)


class OptimizedOnOff(Client):
    def __init__(self, options, classes, controller):
        super().__init__(options, classes)
        self.shadow_data = load_data(
            options,
            [options.adversaries['from']],
            backdoor=options.adversaries['type'].find('backdoor') >= 0
        )
        self.shadow_data['dataloader'].dataset.targets[:] = \
            options.adversaries['to']
        controller.add_sybil(self)
        self.attacking = False

    def switch_mode(self):
        temp = self.data
        self.data = self.shadow_data
        self.shadow_data = temp
        self.attacking = not self.attacking

    def fit(self, verbose=False):
        scale_up = 1
        if self.attacking and self.options.adversaries['scale_up']:
            batch_size = self.options.model_params['batch_size']
            scale_up = (batch_size * self.options.users) / batch_size
        return super().fit(scaling=scale_up, verbose=verbose)


def load_adversary(options, controller):
    """Load the class of the specified adversary"""
    if options.adversaries['optimized']:
        return lambda o, c: OptimizedOnOff(o, c, controller)
    return OnOff

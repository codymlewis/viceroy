"""
Server that controls sybils
"""

import random

import torch

from server.global_model import load_fit_fun
import utils
from utils.datasets import load_data
from utils.models import load_model


class Controller:
    def __init__(self, options):
        self.options = options
        self.setup()

    def setup(self):
        self.sybils = []
        self.alpha = torch.tensor([])
        self.grads = torch.tensor([])
        self.fit_fun = load_fit_fun(self.options.fit_fun)
        self.params = self.options.model_params
        self.histories = dict()
        self.net = None
        self.epochs = 0
        self.delay = self.options.adversaries['delay']
        self.toggle_record = []
        self.lr = self.params['learning_rate'][0]
        self.learning_rates = self.params['learning_rate'].copy()
        del self.learning_rates[0]
        self.lr_changes = self.params['lr_changes'].copy()
        self.attack = False

    def add_sybil(self, sybil):
        self.sybils.append(sybil)

    def intercept(self, grads):
        self.epochs += 1
        if self.epochs > self.delay:
            with torch.no_grad():
                prev_hist = {k: v.detach().clone() for k, v in self.histories.items()}
                max_alpha = 1 / self.options.users
                alpha = self.fit_fun(self, grads, self.options.params)
                avg_syb_alpha = alpha[-len(self.sybils):].mean()
                p = self.attack and \
                    avg_syb_alpha < self.options.adversaries['beta'] * max_alpha
                q = not self.attack and \
                    avg_syb_alpha > self.options.adversaries['gamma'] * max_alpha
                if p or q:
                    self.toggle_record.append(self.epochs)
                    self.attack = not self.attack
                    for sybil in self.sybils:
                        sybil.switch_mode()
                self.histories = prev_hist
        return self.attack

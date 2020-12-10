"""
Server that controls sybils
"""

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
        self.toggle_time = self.options.adversaries['delay']
        self.toggle_record = []
        self.lr = self.params['learning_rate'][0]
        self.learning_rates = self.params['learning_rate'].copy()
        del self.learning_rates[0]
        self.lr_changes = self.params['lr_changes'].copy()
        # Train adversarial model
        data = load_data(self.options, train=True, shuffle=True)
        data['dataloader'].dataset.targets[:] = self.options.adversaries['to']
        model = load_model(self.options.model_params)
        model.fit(
            data['dataloader'],
            epochs=5,
            verbose=self.options.verbosity > 1
        )
        with torch.no_grad():
            self.goal_model = utils.flatten_params(model.get_params(), self.params)
        self.attack = False

    def add_sybil(self, sybil):
        self.sybils.append(sybil)

    def intercept(self, net_params, grads):
        self.epochs += 1
        if self.lr_changes and self.epochs > self.lr_changes[0]:
            self.lr = self.learning_rates[0]
            del self.learning_rates[0]
            del self.lr_changes[0]
        with torch.no_grad():
            new_grads = torch.tensor([], device=self.params['device'])
            for grad in grads:
                new_grad = torch.tensor([], device=self.params['device'])
                for p in grad['params']:
                    new_grad = torch.cat((new_grad, p.flatten()))
                new_grads = torch.cat((new_grads, new_grad.unsqueeze(0))) *\
                    self.epochs
            if len(self.grads) == 0:
                self.grads = new_grads
            else:
                self.grads += new_grads
            if len(self.alpha) == 0:
                self.alpha = self.fit_fun(self, grads, self.options.params)
            else:
                self.alpha += self.fit_fun(self, grads, self.options.params)
        if self.epochs == self.toggle_time:
            self.toggle_record.append(self.epochs)
            self.attack = not self.attack
            for sybil in self.sybils:
                sybil.switch_mode()
            # TODO: remove double fitting
            grads[-len(self.sybils):] = [s.fit()[1] for s in self.sybils]
            with torch.no_grad():
                current_model = utils.flatten_params(net_params, self.params)
                if self.attack:
                    self.normal_model = current_model
                self.toggle_time += self.mde(
                    current_model,
                    self.lr,
                    self.options.adversaries['optimized_look_ahead']
                )
                # print()
                # print(self.toggle_time)
                # print()

    def mde(self, current_model, eta, tau):
        with torch.no_grad():
            min_sim = 1
            result = 1
            model = current_model
            expanded_alpha = (self.alpha / self.epochs).expand_as(
                self.grads.T).T.to(self.options.model_params['device'])
            lr_changes = self.lr_changes.copy()
            lrs = self.learning_rates.copy()
            if self.epochs + tau > self.options.server_epochs:
                tau = self.options.server_epochs + 1 - self.epochs
            for i in range(1, tau + 1):
                if lr_changes and self.epochs + i > lr_changes[0]:
                    eta = lrs[0]
                    del lr_changes[0]
                    del lrs[0]
                model -= (eta * expanded_alpha * (self.grads / self.epochs)).sum(dim=0)
                sim = torch.cosine_similarity(
                    self.goal_model if self.attack else self.normal_model,
                    model,
                    dim=-1
                )
                if sim < min_sim:
                    result = i
                    min_sim = sim
            return result

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
        self.sybils = []
        self.alpha = torch.tensor([])
        self.grads = torch.tensor([])
        self.fit_fun = load_fit_fun(options.fit_fun)
        self.params = options.model_params
        self.histories = dict()
        self.net = None
        self.epochs = 0
        self.toggle_time = options.adversaries['delay']
        print(self.toggle_time)
        # Train adversarial model
        data = load_data(options, train=True, shuffle=True)
        data['dataloader'].dataset.targets[:] = options.adversaries['to']
        model = load_model(options.model_params)
        model.fit(data['dataloader'], epochs=5)
        with torch.no_grad():
            self.goal_model = utils.flatten_params(model.get_params(), self.params)

    def add_sybil(self, sybil):
        self.sybils.append(sybil)

    def intercept(self, net_params, grads):
        self.epochs += 1
        if self.epochs == self.toggle_time:
            for sybil in self.sybils:
                sybil.switch_mode()
            # TODO: remove double fitting
            grads[-len(self.sybils):] = [s.fit()[1] for s in self.sybils]
            with torch.no_grad():
                new_grads = torch.tensor([], device=self.params['device'])
                for grad in grads:
                    new_grad = torch.tensor([], device=self.params['device'])
                    for p in grad['params']:
                        new_grad = torch.cat((new_grad, p.flatten()))
                    new_grads = torch.cat((new_grads, new_grad.unsqueeze(0)))
                if len(self.grads) == 0:
                    self.grads = new_grads
                else:
                    self.grads += new_grads
                if len(self.alpha) == 0:
                    self.alpha = self.fit_fun(self, grads, self.options.params)
                else:
                    self.alpha += self.fit_fun(self, grads, self.options.params)
                current_model = utils.flatten_params(net_params, self.params)
                self.toggle_time += mde(
                    self.goal_model,
                    current_model,
                    self.grads / self.epochs,
                    self.params['learning_rate'][0],
                    self.alpha / self.epochs,
                    5
                )
                print(self.toggle_time)
                print()

def mde(target_model, current_model, grads, eta, alpha, tau_max):
    result = 1
    best_sim = 0
    for tau in range(1, tau_max):
        calc = mde_calc(current_model, grads, eta, alpha, tau)
        # TODO: reconstruct models, calculate loss between the predictions
        if (cs := torch.cosine_similarity(calc, target_model, dim=0)) > best_sim:
            best_sim = cs
            result = tau
    return result


def mde_calc(current_model, grads, eta, alpha, tau):
    result = current_model
    for i in range(tau):
        result *= 1 - (eta * alpha).sum()
    for i in range(tau):
        for u in range(len(alpha)):
            intermediate = grads[u] * eta * alpha[u]
            for j in range(tau - i):
                intermediate *= 1 - alpha[u]  # [t + j + 1]
            result += intermediate
    return result


    # TODO: reset after each sim
    # TODO: dynamic learning rate

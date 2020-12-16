"""
A model for ML Models and a function to load them

Author: Cody Lewis
"""

from abc import abstractmethod

import torch.nn as nn
import torch.optim as optim
import torchvision

import utils.errors


class Model(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.lr = params['learning_rate'][0]
        self.learning_rates = params['learning_rate'].copy()
        del self.learning_rates[0]
        self.lr_changes = params['lr_changes'].copy()
        self.epoch_count = 0

    @abstractmethod
    def forward(self, *x):
        pass

    def fit(self, data, epochs=1, scaling=1, verbose=True):
        """
        Fit the model for some epochs, return history of loss values and the
        gradients of the changed parameters

        Keyword arguments:
        x -- training data
        y -- training labels
        epochs -- number of epochs to train for
        verbose -- output training stats if True
        """
        optimizer = optim.SGD(
            self.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.0001
        )
        criterion = nn.CrossEntropyLoss()
        data_count = 0
        for i in range(epochs):
            optimizer.zero_grad()
            x, y = next(iter(data))
            x = x.to(self.params['device'])
            y = y.to(self.params['device'])
            output = self(x)
            loss = criterion(output, y)
            if verbose:
                print(
                    f"Epoch {i + 1}/{epochs} loss: {loss}",
                    end="\r"
                )
            loss.backward()
            optimizer.step()
            data_count += len(y)
        self.epoch_count += 1
        if self.lr_changes and self.epoch_count > self.lr_changes[0]:
            self.lr = self.learning_rates[0]
            del self.learning_rates[0]
            del self.lr_changes[0]
        if verbose:
            print()
        return loss, {
            "params": [scaling * -self.lr * p.grad for p in self.parameters()],
            "data_count": data_count
        }

    def get_params(self):
        """Get the tensor form parameters of this model"""
        return [p.data for p in self.parameters()]

    def copy_params(self, params):
        """Copy input parameters into self"""
        for p, t in zip(params, self.parameters()):
            t.data.copy_(p)


class SoftMaxModel(Model):
    """The softmax perceptron class"""
    def __init__(self, params):
        super().__init__(params)
        self.features = nn.ModuleList([
            nn.Linear(
                params['num_in'], params['num_in'] * params['params_mul']
            ),
            nn.Sigmoid(),
            nn.Linear(
                params['num_in'] * params['params_mul'], params['num_out']
            ),
            nn.Softmax(dim=1)
        ]).eval()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x


class SqueezeNet(Model):
    """The SqueezeNet DNN Class"""
    def __init__(self, params):
        super().__init__(params)
        net = torchvision.models.__dict__["squeezenet1_1"](pretrained=True)
        net.classifier[1] = nn.Conv2d(
            512, params['num_out'], kernel_size=(1, 1), stride=(1, 1)
        )
        self.features = nn.ModuleList(
            [f for f in net.features] +
            [f for f in net.classifier]
        ).eval()
        super().copy_params([p.data for p in net.parameters()])

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x.flatten(1)


def load_model(params):
    """Load the model specified in params"""
    models = {
        "softmax": SoftMaxModel,
        "squeeze": SqueezeNet,
    }
    model_name = params['architecture']
    if (chosen_model := models.get(model_name)) is None:
        raise utils.errors.MisconfigurationError(
            f"Model '{model_name}' does not exist, " +
            f"possible options: {set(models.keys())}"
        )
    return chosen_model(params).to(params['device'])

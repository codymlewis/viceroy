"""
Pytorch implementation of a softmax perceptron

Author: Cody Lewis
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time


class Model(nn.Module):
    def __init__(self, lr, lr_changes):
        super().__init__()
        self.lr = lr[0]
        self.learning_rates = lr.copy()
        del self.learning_rates[0]
        self.lr_changes = lr_changes.copy()
        self.epoch_count = 0

    def fit(self, x, y, batch_size=0, epochs=1, verbose=True):
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
        if isinstance(x, torch.utils.data.dataset.Dataset):
            n = len(x)
        else:
            n, _ = x.shape
        print(f"n: {n}")
        data_count = 0
        for i in range(epochs):
            optimizer.zero_grad()
            if 0 < batch_size < n:
                ids = torch.randperm(n)[:batch_size]
                output = self(x[ids])
                if len(output.shape) > 2:
                    output = output.flatten(1)
                print(output.shape)
                # sample_x = x[ids]
                # output = torch.tensor([self(x[i]) for i in ids])
                sample_y = y[0][ids]
                data_count += batch_size
            else:
                # sample_x = x
                output = torch.tensor([self(x_s) for x_s in x])
                sample_y = y[0]
                data_count = n
            # output = self(sample_x)
            loss = criterion(output, sample_y)
            if verbose:
                print(
                    f"Epoch {i + 1}/{epochs} loss: {loss}",
                    end="\r"
                )
            loss.backward()
            optimizer.step()
        self.epoch_count += 1
        if self.lr_changes and self.epoch_count > self.lr_changes[0]:
            self.lr = self.learning_rates[0]
            del self.learning_rates[0]
            del self.lr_changes[0]
        if verbose:
            print()
        return loss, {
            "params": [-self.lr * p.grad for p in self.parameters()],
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
    def __init__(self, num_in, num_out, lr=[0.01], lr_changes=[], params_mul=10):
        super().__init__(lr, lr_changes)
        self.features = nn.ModuleList([
            nn.Linear(num_in, num_in * params_mul),
            nn.Sigmoid(),
            nn.Linear(num_in * params_mul, num_out),
            nn.Softmax(dim=1)
        ]).eval()

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x


class SqueezeNet(Model):
    def __init__(self, num_in, num_out, lr=[0.01], lr_changes=[], params_mul=10):
        super().__init__(lr, lr_changes)
        net = torchvision.models.__dict__["squeezenet1_1"](pretrained=True)
        net.classifier[1] = nn.Conv2d(
            512, num_out, kernel_size=(1, 1), stride=(1, 1)
        )
        self.features = nn.ModuleList(
            [f for f in net.features] +
            [f for f in net.classifier]
        ).eval()
        super().copy_params([p.data for p in net.parameters()])

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x


MODELS = {
    "softmax": SoftMaxModel,
    "squeeze": SqueezeNet,
}

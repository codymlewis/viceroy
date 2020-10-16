"""
Pytorch implementation of a softmax perceptron

Author: Cody Lewis
"""

import torch
import torch.nn as nn
import torch.optim as optim


class SoftMaxModel(nn.Module):
    """The softmax perceptron class"""
    def __init__(self, num_in, num_out, lr=0.01):
        super(SoftMaxModel, self).__init__()

        self.features = nn.ModuleList([
            nn.Linear(num_in, num_out),
            nn.Softmax(dim=1)
        ]).eval()
        self.lr = lr

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x

    def fit(self, x, y, epochs, verbose=True):
        """
        Fit the model for some epochs, return history of loss values and the
        gradients of the changed parameters

        Keyword arguments:
        x -- training data
        y -- training labels
        epochs -- number of epochs to train for
        verbose -- output training stats if True
        """
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.BCELoss()
        history = {'loss': []}
        for i in range(epochs):
            optimizer.zero_grad()
            output = self(x)
            history['loss'].append(criterion(output, y))
            if verbose:
                print(
                    f"Epoch {i + 1}/{epochs} loss: {history['loss'][-1]}",
                    end="\r"
                )
            history['loss'][-1].backward()
            optimizer.step()
        if verbose:
            print()
        return history, {
            "params": [p.grad for p in self.parameters()],
            "data_count": len(x)
        }

    def get_params(self):
        """Get the tensor form parameters of this model"""
        return [p.data for p in self.parameters()]

    def copy_params(self, params):
        """Copy input parameters into self"""
        for p, t in zip(params, self.parameters()):
            t.data.copy_(p)

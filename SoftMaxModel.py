"""
Pytorch implementation of a softmax perceptron

Author: Cody Lewis
"""

import torch
import torch.nn as nn
import torch.optim as optim

import utils


class SoftMaxModel(nn.Module):
    """The softmax perceptron class"""
    def __init__(self, num_in, num_out, lr=0.01):
        super(SoftMaxModel, self).__init__()

        self.features = nn.ModuleList([
            nn.Linear(num_in, num_in * 10),
            nn.Sigmoid(),
            nn.Linear(num_in * 10, num_out),
            nn.Softmax(dim=1)
        ]).eval()
        self.lr = lr

    def forward(self, x):
        for feature in self.features:
            x = feature(x)
        return x

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
        optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        n, _ = x.shape
        for i in range(epochs):
            optimizer.zero_grad()
            if 0 < batch_size < n:
                ids = torch.randperm(n)[:batch_size]
                sample_x = x[ids]
                sample_y = y[0][ids]
            else:
                sample_x = x
                sample_y = y
            output = self(sample_x)
            loss = criterion(output, sample_y)
            if verbose:
                print(
                    f"Epoch {i + 1}/{epochs} loss: {loss}",
                    end="\r"
                )
            loss.backward()
            optimizer.step()
        if verbose:
            print()
        return loss, {
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


if __name__ == '__main__':
    data = utils.load_data("mnist")
    net = SoftMaxModel(data['x_dim'], data['y_dim'])
    net.fit(data['x'], data['y'], 64, 50000)

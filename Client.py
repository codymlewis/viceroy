"""
Classes and functions for the client networking aspect of federated learning

Author: Cody Lewis
"""

import socket
import pickle

import torch
import torch.nn as nn

import SoftMaxModel
import utils


class Client:
    def __init__(self, x, y):
        dims = utils.get_dims(x.shape, y.shape)
        self.net = SoftMaxModel.SoftMaxModel(dims['x'], dims['y'])
        self.x = x
        self.y = y
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect(self, host, port):
        """Connect to the host:port federated learning server"""
        self.socket.connect((host, port))

    def fit(self):
        criterion = nn.BCELoss()
        e = 0
        while (msg := self.socket.recv(4096)) != b'DONE':
            self.net.copy_params(pickle.loads(msg))
            e += 1
            history, grads = self.net.fit(self.x, self.y, 1, verbose=False)
            self.socket.sendall(pickle.dumps(grads))
            print(
                f"Epoch: {e}, Loss: {criterion(self.net(X), Y)}",
                end="\r"
            )
            # An improvement would be to save grads as a backlog and concurrent
            # send them when the server is ready
        print()


if __name__ == '__main__':
    # X = torch.tensor([
    #     [0, 0],
    #     [0, 1],
    #     [1, 0],
    #     [1, 1]
    # ], dtype=torch.float32)
    # Y = torch.tensor([
    #     [1, 0],
    #     [1, 0],
    #     [1, 0],
    #     [0, 1]
    # ], dtype=torch.float32)
    X, Y = utils.load_data("mnist")
    client = Client(X, Y)
    HOST, PORT = '127.0.0.1', 5000
    print(f"Connecting to {HOST}:{PORT}")
    client.connect(HOST, PORT)
    client.fit()

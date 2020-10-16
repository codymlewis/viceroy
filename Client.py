"""
Classes and functions for the client networking aspect of federated learning

Author: Cody Lewis
"""

import socket
import pickle

import torch
import torch.nn as nn

import SoftMaxModel


class Client:
    def __init__(self, x, y):
        self.net = SoftMaxModel.SoftMaxModel(len(x[0]), len(y[0]))
        self.x = x
        self.y = y
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def connect(self, host, port):
        """Connect to the host:port federated learning server"""
        self.socket.connect((host, port))
        self.net.copy_params(pickle.loads(self.socket.recv(1024)))

    def fit(self):
        if self.socket.recv(1024) != b'OK':
            pass
        criterion = nn.BCELoss()
        e = 0
        while True:
            e += 1
            history, grads = self.net.fit(self.x, self.y, 1, verbose=False)
            self.socket.sendall(pickle.dumps(grads))
            print(
                f"Epoch: {e}, Loss: {criterion(self.net(X), Y)}",
                end="\r"
            )
            if self.socket.recv(1024) != b'OK':
                break
            # An improvement would be to save grads as a backlog and concurrent
            # send them when the server is ready
        print()


if __name__ == '__main__':
    X = torch.tensor([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=torch.float32)
    Y = torch.tensor([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1]
    ], dtype=torch.float32)
    client = Client(X, Y)
    HOST, PORT = '127.0.0.1', 5000
    print(f"Connecting to {HOST}:{PORT}")
    client.connect(HOST, PORT)
    client.fit()

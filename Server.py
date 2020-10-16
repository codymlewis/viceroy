"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""

import socket
import pickle
import torch
import torch.nn as nn

import GlobalModel


class Server:
    """Federated learning server class"""
    def __init__(self, num_in, num_out, port=5000):
        self.net = GlobalModel.GlobalModel(num_in, num_out)
        self.num_clients = 0
        self.address = ('', port)
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(self.address)
        self.clients = []

    def accept_client(self, s):
        """Accept a client and update the model accordingly"""
        res = s.accept()
        self.num_clients += 1
        self.net.add_client()
        return res

    def accept_clients(self, num_clients):
        """Accept some clients to the system"""
        self.socket.listen(num_clients)
        self.clients.extend([
            (c, addr) for c, addr in [
                self.accept_client(self.socket) for _ in range(num_clients)
            ]
        ])
        for c, _ in self.clients:
            c.send(pickle.dumps(self.net.get_params()))

    def fit(self, X, Y, epochs):
        criterion = nn.BCELoss()
        for e in range(epochs):
            grads = dict()
            for i, (c, _) in enumerate(self.clients):
                c.send(b'OK')
                grads[i] = pickle.loads(c.recv(4096))
            self.net.fit(1, grads)
            print(
                f"Epoch: {e + 1}/{epochs}, Loss: {criterion(server.net.predict(X), Y)}",
                end="\r"
            )
        print()

    def close(self):
        for c, _ in self.clients:
            c.close()
        self.clients = []



if __name__ == '__main__':
    PORT = 5000
    server = Server(2, 2, PORT)
    print(f"Starting server on port {PORT}")
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
    server.accept_clients(2)
    server.fit(X, Y, 5000)
    server.close()

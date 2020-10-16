"""
Classes and functions for the server networking aspect of federated learning

Author: Cody Lewis
"""

import socket

import GlobalModel


class Server:
    """Federated learning server class"""
    def __init__(self, num_in, num_out):
        self.net = GlobalModel.GlobalModel(num_in, num_out)
        self.num_clients = 0
        self.address = ('', 5000)

    def accept_client(self, s):
        """Accept a client and update the model accordingly"""
        res = s.accept()
        self.num_clients += 1
        self.net.add_client()
        return res

    def accept_clients(self, num_clients):
        """Accept some clients to the system"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind(self.address)
            s.listen(num_clients)
            clients = [
                (c, addr) for c, addr in [
                    self.accept_client(s) for _ in range(num_clients)
                ]
            ]
            for c, addr in clients:
                msg = c.recv(1024)
                print(f"{addr} >> {msg}")
                c.send(msg)
                c.close()


if __name__ == '__main__':
    server = Server(2, 2)
    server.accept_clients(2)

import socket

import SoftMaxModel


class Client:
    def __init__(self, num_in, num_out):
        self.net = SoftMaxModel.SoftMaxModel(num_in, num_out)

    def connect(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.connect((host, port))
            s.sendall(b'Hello, there')
            print(s.recv(1024))


if __name__ == '__main__':
    client = Client(2, 2)
    client.connect('127.0.0.1', 5000)

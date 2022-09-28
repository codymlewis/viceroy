"""
Defines the network architecture for the FL system.
"""


import numpy as  np
import optax

import fl.utils.functions


class Controller:
    """
    Holds a collection of clients and connects to other Controllers.  
    Handles the update step of each of the clients and passes the respective gradients
    up the chain.
    """
    def __init__(self, C):
        """
        Construct the Controller.

        Arguments:
        - C: percent of clients to randomly select for training at each round
        """
        self.clients = []
        self.switches = []
        self.C = C
        self.K = 0
        self.update_transform_chain = []

    def __len__(self):
        return len(self.clients) + sum([len(s) for s in self.switches])

    def add_client(self, client):
        """Connect a client directly to this controller"""
        self.clients.append(client)
        self.K += 1

    def add_switch(self, switch):
        """Connect another controller (referred to as switch) to this controller"""
        self.switches.append(switch)
    
    def add_update_transform(self, update_transform):
        """Add a function that transforms the updates before passing them up the chain"""
        self.update_transform_chain.append(update_transform)

    def __call__(self, params, rng=np.random.default_rng(), return_weights=False):
        """
        Update each connected client and return the generated update. Recursively call in connected controllers
        
        Arguments:
        - params: the parameters of the global model from the most recent round
        - rng: the random number generator to use
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        all_updates = []
        for switch in self.switches:
            all_updates.extend(switch(params, rng, return_weights))
        idx = rng.choice(self.K, size=int(self.C * self.K), replace=False)
        for i in idx:
            p = params
            sum_grads = None
            for _ in range(self.clients[i].epochs):
                grads, self.clients[i].opt_state, updates = self.clients[i].update(p, self.clients[i].opt_state, *next(self.clients[i].data))
                p = optax.apply_updates(p, updates)
                sum_grads = grads if sum_grads is None else fl.utils.functions.tree_add(sum_grads, grads)
            all_updates.append(p if return_weights else sum_grads)
        return fl.utils.functions.chain(self.update_transform_chain, all_updates)


class Network:
    """Higher level class for tracking each controller and client"""
    def __init__(self, C=1.0):
        """Construct the Network.

        Arguments:
        - C: percent of clients to randomly select for training at each round
        """
        self.clients = []
        self.controllers = {}
        self.server_name = ""
        self.C = C

    def __len__(self):
        """Get the number of clients in the network"""
        return len(self.clients)

    def add_controller(self, name, server=False):
        """Add a new controller with name into this network"""
        self.controllers[name] = Controller(self.C)
        if server:
            self.server_name = name
    
    def get_controller(self, name):
        """Get the controller with the specified name"""
        return self.controllers[name]

    def add_host(self, controller_name, client):
        """Add a client to the specified controller in this network"""
        self.clients.append(client)
        self.controllers[controller_name].add_client(client)

    def connect_controllers(self, from_con, to_con):
        """Connect two controllers in this network"""
        self.controllers[from_con].add_switch(self.controllers[to_con])

    def __call__(self, params, rng=np.random.default_rng(), return_weights=False):
        """
        Perform an update step across the network and return the respective updates

        Arguments:
        - params: the parameters of the global model from the most recent round
        - rng: the random number generator to use
        - return_weights: if True, return the weights of the clients else return the gradients from the local training
        """
        return self.controllers[self.server_name](params, rng, return_weights)

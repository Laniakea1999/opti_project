import numpy as np
from random import sample

import strategy.strategy as strat

# Proportion of neighbors to sample to then broadcast data to
c = 0.5

class Node():
    """
    A node in the simulated network.
    Nodes can receive data, process it,
    and broadcast data to other nodes.
    Received data is put in the inbox,
    and data to be broadcast is put in the outbox.
    """

    def __init__(self, uid, net):
        self.uid = uid
        self.neighbors = set()
        self.network = net

        self.X_train = None # Local dataset
        self.y_train = None
        #self.X_test = None
        #self.y_test = None
        self.params = None # Local parameters
        self.loss = np.inf

        self.inbox = []
        self.outbox = []


    def add_neighbor(self, neighbor_id):
        """
        Notify this node of a new neighbor
        """
        self.neighbors.add(neighbor_id)

    def broadcast(self):
        """
        Broadcast parameters to a subset of neighboring
        nodes
        """
        # 1. Pick a subset of neighbors
        # 2. Send data to neighbors in subset

        k = int(np.ceil(len(self.neighbors) * c))
        subset = sample(self.neighbors, k)

        for data in self.outbox:
            for n in subset:
                self.network.send(self.uid, n, data)
        
        self.outbox = []

    def receive(self, data):
        """
        Receive model parameters into this node's inbox
        """
        self.inbox.append(data)

    def process(self):
        """
        Build a single model from the parameters in the inbox,
        perform some iterations of gradient descent,
        and store the updated parameters in the outbox.
        """
        loss, params = strat.compute(self.X_train, self.y_train, self.network.X_test, self.network.y_test, strat.aggregate(self.params, self.inbox))
        self.params = params
        self.loss = loss

        self.inbox = []
        self.outbox.append(self.params)


    def __repr__(self):
        return "{uid} -> {n}".format(uid=self.uid, n=self.neighbors)
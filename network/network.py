
import sys
from fileinput import FileInput

import numpy as np
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import strategy.strategy as strat
from network.node import Node
from helpers import *

class Network():
    """
    # Network class
    Represents a simulated network, with no particular
    topology, composed of nodes.
    Network communications happen through this class,
    to keep link state and possibly implement bandwidth/latency issues.
    """

    def __init__(self, nodes, links):
        self.nodes = nodes
        self.links = links

        # For stat calculations 
        self.n_iter = 0
        self.best_losses = []
        self.worst_losses = []
        self.avg_losses = []

    def singleton():
        """
        A network with a single node.
        """
        net = Network({}, {})
        net.nodes.update({0: Node(0, net)})
        print_success("Created single-node network")
        return net


    def from_stream(file_path='-', delimiter=','):
        """
        Load the network from a file or stdin, line by line, following this format:
        ```
        0:\tn
        1:\tnode1,node2,failure_probability
        2:\t...
        ```
        where `n` is the number of edges, given by the first line,
        `node1` and `node2` are node IDs, `failure_probability` is
        the probability of a given message to be dropped over the link.
        """

        nodes = {}
        links = {}

        net = Network(nodes, links)

        with FileInput(file_path) as f:
            # First line is the number of edges
            n = int(f.readline())
            for i in range(n):
                data = f.readline().split(delimiter)
                u, v = int(data[0]), int(data[1])
                f_prob = float(data[2]) if (len(data) > 2) else 0.0

                smaller = min(u, v)
                bigger = max(u, v)

                for node in [u, v]:
                    if (net.nodes.get(node) == None):
                        net.nodes.update({node: Node(node, net)})

                net.nodes[u].add_neighbor(v)
                net.nodes[v].add_neighbor(u)
                    
                net.links.update({(smaller, bigger): f_prob})

        print_success("Done loading network. \t\t[{n} vertices, {l} edges]".format(n=len(nodes), l=len(links)))

        return net



    def set_data(self, X_train, X_test, y_train, y_test, node_test_size=0.5):
        """
        Sets the dataset this network will work on, and
        distributes the training dataset over all nodes.
        """
        n = len(self.nodes)

        if (n == 0):
            print_error("Attempted to distribute data to empty graph")
            sys.exit(2)
        else:
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test

            X_partitions = np.array_split(self.X_train, n)
            y_partitions = np.array_split(self.y_train, n)


            for i, node in enumerate(self.nodes.keys()):

                self.nodes[node].X_train = X_partitions[i]
                self.nodes[node].y_train = y_partitions[i]
            
            print_success("Distributed data to all nodes.\t[~{} rows per node]".format(len(X_partitions[0])))


    def send(self, src, to, data):
        """
        Send a message to a node over the network.
        """
        # For now, we omit packet dropping
        # Also no need for a search algorithm
        # as the network is considered static
        # and nodes only communicate with their neighbors
        self.nodes[to].receive(data)


    def iterate(self):
        """
        Perform a single network-wide iteration:
            - All nodes perform their computations (see `Node.process()`)
            - All nodes then broadcast their updated parameters to a subset of their neighbors
        """
        for n, node in self.nodes.items():
            node.process()

        for n, node in self.nodes.items():
            node.broadcast()



    def train_weights(self, max_iterations, threshold):
        """
        Perform iterations over the network until either
        the max number of iterations is reached or until the differences
        in losses between epochs is smaller than a threshold
        """

        print("\nTraining weights over network...")
        i = 0
        with tqdm(total=max_iterations) as pbar:
            converged = False

            while (i < max_iterations and not converged):
                self.iterate()
                pbar.update(1)
                i += 1
                loss, _ = self.get_best_candidate()
                self.best_losses.append(loss)
                self.worst_losses.append(np.max([n.loss for n in self.nodes.values()]))
                self.avg_losses.append(np.average([n.loss for n in self.nodes.values()]))
                converged = loss < threshold
                
        self.n_iter = i
        print()
        print_success(f"Model has been trained.\t{f'[Converged after {i} iterations]' if converged else '[Max number of iterations reached]'}")

    def get_best_candidate(self):
        best_loss = np.inf
        best_candidate = 0
        for node in self.nodes.values():
            if (node.loss < best_loss):
                best_candidate = node.uid
                best_loss = node.loss
        return best_loss, best_candidate



    def build_model(self, node=None):
        """
        Selects a random node in the network,
        aggregates parameters from both itself and its neighbors,
        and builds a network from these parameters.
        """
        if (node != None):
            center = node
        else:
            _, center = self.get_best_candidate()
        clf = strat.reconstruct(self.nodes[center].params)
        return clf



    def __repr__(self):
        s = 'Nodes: {n}\tEdges: {e}\n\n'.format(n=len(self.nodes), e=len(self.links.keys()))
        for nodes, f_prob in self.links.items():
            s += '{u} -> {v}\t\tf_prob: {f_prob}\n'.format(u=nodes[0], v=nodes[1], f_prob=f_prob)
        return s

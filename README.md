# Effects of topology on decentralized classification

> Zied Mustapha, Kepler Warrington-Arroyo

> Project done as part of the EPFL CS439 course given by Professors Martin Jaggi & Nicolas Flammarion

## Description
This project aims to shed light on the effect of particular network topology on a decentralized classification task. The code in this repository serves to perform such a task on a simulated network, and provide insight on the convergence of the overall model and the performance of intermediate models found on nodes.

## Requirements
```
scikit-learn
numpy
matplotlib
seaborn
networkx
```

## Main Files
- `sketchbook.ipynb`: a Jupyter Notebook for interactive use
- `main.py`: run with:
```
python3 main.py --dataset ['iris', 'digits', 'wine'] -n number_of_runs < network_file
```
This builds a decentralized model over the network to classify the specified dataset, stopping when either 1000 iterations have been performed or a node has converged. Several runs are performed, each with a different distribution of the data, to compute statistics about the process. The script then generates a report in `/output` which includes some statistics and plots of the best-performing graph and its associated confusion matrix on the test set.
> Note 1: in some cases, the dataset + topology combination does not allow all nodes to have sufficient data to perform the correct classification. In this case the program will quit early with a Python error.
>> Re-running it several times may yield a successful attempt, but in most cases it is best to reduce the number of nodes in the dataset to get a higher chance of success.

> Note 2: The `--type 'output'` is untested, but if provided, requires  `-X` and `-y` arguments indicating the path of the X and y datasets, as `.csv` files. The X data should already be preprocessed for best results. Neither X nor y should contain headers.


- `gen_graph.py`: run with:
```
python3 gen_graph.py --type graph_type -n number_of_nodes [-c number_of_cities -s number_of_suburbs] > output_file
```
This creates a network of the specified type, with the given parameters – the output can be piped to a file to be saved.
The different graph types are as follows:
- `'ring'`: a graph where every node is of degree 2
- `'full'`: a fully-connected graph
- `'star'`: a star-shaped graph (one node at the center, all other nodes are its neighbor and are of degree 1)
- `'city'`: an extension of a star graph, where 'cities' are connected to each other while 'suburbs' are only connected to cities.

Some pre-generated networks can be found in `/data/topologies`.

## Code structure description
The `network` module contains classes necessary to build the network simulation. A network can be built from a stream (`stdin`, or a file). 
>The first line of the stream must be the number of edges in the network, followed by one edge per line, where each edge $e = (u, v)$ is given as `u,v` in the file.

Each node is represented by the `Node` class. Nodes have their own training data, a set of neighbors, and their own model parameters. They also have in- and outboxes used for communication between nodes. Nodes send data through their associated `Network` object. The `Node` class only contains what is necessary to participate in the simulation: the concrete steps they take at each iteration are implemented in the `strategy` module, and set at runtime.

A Network is composed of nodes and links. It mainly serves as the overarching orchestrator of the simulation, calling upon all nodes to perform their computations, managing communication, and checking the state of the models at each iteration.

The `strategy` module contains functions used by nodes during their computation.
> 'Hollow' versions of these functions are implemented and called upon by default, but do nothing. To specify the exact methods through which nodes should reduce several model parameters into a single instance, perform their computations, and build a model from their local parameters, the default implementations must be redirected to concrete function implementations. This is already done in both `main.py` and `sketchbook.py`.
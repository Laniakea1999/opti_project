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

## Files
- `sketchbook.ipynb`: a Jupyter Notebook for interactive use
- `main.py`: run with:
```
python3 main.py --dataset ['iris', 'digits', 'wine'] -n number_of_runs < network_file
```
This builds a decentralized model over the network to classify the specified dataset, stopping when either 1000 iterations have been performed or a node has converged. Several runs are performed, each with a different distribution of the data, to compute statistics about the process. The script then generates a report in `/output` which includes some statistics and plots of the best-performing graph and its associated confusion matrix on the test set.


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
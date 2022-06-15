import os, sys, getopt, copy
import pathlib


from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn import datasets, metrics
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split

from helpers import *
import strategy.strategy as strat
from network.network import Network


def process_args(argv):

    argv = sys.argv

    long_args = ['dataset=', 'n_runs=']
    short_args = 'd:n:'

    usage_string = "usage:\t{name} --dataset dt_name -n n_runs".format(name=argv[0])

    dt = None
    n_runs = None

    try:
        opts, args = getopt.getopt(argv[1:],short_args,long_args)
    except getopt.GetoptError:
        print(usage_string)
        sys.exit(1)
    
    for opt, arg in opts:
        if opt == '-h':
            print(usage_string)
            sys.exit()

        elif opt in ("-n", "--n_runs"):
            n_runs = int(arg)
        elif opt in ("-d", "--dataset"):
            dt = arg

    if (dt == None):
        print_error("No dataset specified!")
        sys.exit(1)
    elif (n_runs == None):
        print_error("Missing number of runs")
        sys.exit(1)
    else:
        return dt, n_runs




def run(network, X, y, iterations, dir):
    best_nodes = []
    accuracies = []
    confusion_matrices = []
    classification_reports = []
    figures = []

    convergence_iterations = []


    with open(dir + '/report.txt', 'w') as report:

        report.write(f"n_runs:\t{iterations}\n\n")

        for i in range(iterations):
            print()
            report.write(f"[Iteration {i}]\n\n")
            print_metric("Iteration", i)
            print()
            net = copy.deepcopy(network) # So we can restore from the network without reading the graph from stdin

            # Distribute data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.5, shuffle=True
            )

            net.set_data(X_train, X_test, y_train, y_test)

            # Train weights
            classes = np.unique(y)

            strat.aggregate = strat.aggregate_sgd(classes)
            strat.compute = strat.compute_sgd()
            strat.convergence_loss = metrics.mean_squared_error

            strat.reconstruct = strat.reconstruct_sgd(loss='squared_error', alpha=50.0, penalty='l2', max_iter=5)

            net.train_weights(
                max_iterations=1000,
                threshold=0.1
                )

            print_success("Writing results to report...")

            # Save results
            convergence_iterations.append(net.n_iter)
            best_nodes.append(net.get_best_candidate())

            result = net.build_model().predict(X_test)
            accuracies.append(metrics.accuracy_score(y_test, result))
            classification_reports.append(metrics.classification_report(y_test, result, zero_division=0))
            confusion_matrices.append(metrics.confusion_matrix(y_test, result))
            figures.append(draw_graph(net.nodes.values(), net.links))

            # Write results to report
            report.write(f"epochs:\t\t{convergence_iterations[-1]}\n\n")
            report.write(f"Accuracy:\t\t{accuracies[-1]:.4f}\n\n")
            report.write(classification_reports[-1])
            report.write("\n")
            report.write(str(confusion_matrices[-1]))
            report.write("\n\n\n")

        print_success("All runs performed. Writing final objects...")

        # Write some statistics about the runs
        report.write("\n\n\n")
        report.write(f"Avg accuracy:\t{np.average(accuracies):.4f}\t(stddev: {np.std(accuracies):.4f})\n")
        report.write(f"Avg epochs:\t{np.average(convergence_iterations)}\t(stddev: {np.std(convergence_iterations):.4f})\n")

        # Select run with best accuracy and save graph and confusion matrix
        idx = np.argmax(accuracies)
        figures[idx].savefig(dir + '/best_figure.png')
        plt.clf()
        draw_confusion_matrix(confusion_matrices[idx]).savefig(dir + '/best_confmatrix.png')

        print_success("Done.")



if (__name__ == '__main__'):

    # Arguments are dataset and number of runs
    dataset, runs = process_args(sys.argv)

    # Retrieve dataset
    if (dataset == 'iris'):
        dt = datasets.load_iris()
        X = scale(dt.data)
    elif (dataset == 'wine'):
        dt = datasets.load_wine()
        X = scale(dt.data)
    elif (dataset == 'digits'):
        dt = datasets.load_digits()
        X = dt.images.reshape((len(dt.images), -1))

    y = dt.target

    # Create network
    net = Network.from_stream('-')

    # Create directory for run
    file_path = str(pathlib.Path(__file__).parent.resolve())
    dir = file_path + f'/output/run_{dataset}-{runs}_{datetime.now().strftime("%d-%m-%Y_%H-%M-%S")}'
    os.mkdir(dir)

    run(net, X, y, runs, dir)
    sys.exit()



from turtle import pen
import numpy as np
from pkg_resources import parse_requirements
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning

from sklearn.linear_model import LinearRegression, SGDClassifier, LogisticRegression
from sklearn.svm import SVC

import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)



def aggregate(current_params, new_params_list):
    """
    Aggregate all parameters received in the inbox, and
    returning a single instance of parameters.
    """
    pass

def compute(X, y, parameters):
    """
    Perform a computation on data with parameters,
    and return the new resulting parameters.
    """
    pass

def reconstruct(parameters):
    """
    Build a sklearn model out of parameters found on a node,
    or aggregated from several nodes.
    """
    pass

def convergence_loss(metric):
    """
    Loss metric used to evaluated convergence of the model.
    """
    pass


"""

                LOGISTIC REGRESSION

"""


def reconstruct_logreg(C=100.0, penalty="l2", solver="saga", tol=0.1, max_iter=5):
    def recon(params):
        clf = LogisticRegression(C=C, penalty=penalty, solver=solver, tol=tol, max_iter=max_iter)
        if (params != None):
            clf.classes_ = params[0]
            clf.coef_ = params[1]
            clf.intercept_ = params[2]
        return clf
    return recon


def compute_logreg():

    def logreg(X, y, params):
        clf = reconstruct_logreg()(params)
        
        clf.fit(X, y)

        return [clf.classes_, clf.coef_, clf.intercept_]

    return logreg



def aggregate_logreg(classes):

    n_classes = len(classes)

    def agg(current_params, new_params_list):
        if (current_params == None and new_params_list == []):
            return None

        # Coefficients are (n_classes, n_features) matrices
        coefs = np.zeros(shape=(n_classes, len(current_params[1][0, :])))
        # Intercepts are a (n_classes, ) vector
        intercepts = np.zeros(shape=(n_classes, 1))

        new_params_list.append(current_params)

        n = len(new_params_list)

        for param in new_params_list:
            for idx, c in enumerate(param[0]):
                # Coefs: add c
                coefs[c, :] += param[1][idx, :]
                # Intercept: add itc value to the correct column
                intercepts[c] += param[2][idx]

        return [classes, coefs / n, intercepts.reshape(n_classes, ) / n]

    return agg

        


"""

            STOCHASTIC GRADIENT DESCENT

"""

def reconstruct_sgd(loss='hinge', alpha=100.0, penalty="l2", learning_rate='optimal', tol=0.1, max_iter=5):
    """
    Creates a `SGDClassifier` builder function from the set of model hyperparameters.
    The builder function can then instantiate an `SGDClassifier` from a list of parameters
    obtained through `aggregate_sgd()`.
    """
    def recon(params):
        clf = SGDClassifier(loss=loss, alpha=alpha, penalty=penalty, learning_rate=learning_rate, tol=tol, max_iter=max_iter)
        if (params != None):
            clf.loss_function_ = params[0]
            clf.classes_ = params[1]
            clf.coef_ = params[2]
            clf.intercept_ = params[3]
        return clf
    return recon


def compute_sgd():
    """
    Returns a function which builds a model from
    the node parameters, runs a few iterations of SGD on
    them, and returns the loss and updated parameters. 
    """

    def sgd(X_train, y_train, X_test, y_test, params):
        clf = reconstruct_sgd()(params)
        
        clf.fit(X_train, y_train)

        loss = convergence_loss(y_test, clf.predict(X_test))

        return (loss, [clf.loss_function_, clf.classes_, clf.coef_, clf.intercept_])

    return sgd



def aggregate_sgd(classes):
    """
    Aggregate several instances of parameters of an `SGDClassifier` into a single
    instance of parameters.
    The `classes` argument must provide the labels of all classes in the dataset
    so the function can correctly aggregate parameters across nodes which may not
    have samples of every label.
    """

    n_classes = len(classes)

    def agg(current_params, new_params_list):
        if (current_params == None and new_params_list == []):
            return None

        loss_func = current_params[0]

        # Coefficients are (n_classes, n_features) matrices
        coefs = np.zeros(shape=(n_classes, len(current_params[2][0, :])))
        # Intercepts are a (n_classes, ) vector
        intercepts = np.zeros(shape=(n_classes, ))

        new_params_list.append(current_params)

        n = len(new_params_list)

        for param in new_params_list:
            if (n_classes == 2): # Binary classification
                coefs += param[2]
                intercepts += param[3]
            else: # Multiclass classification
                for idx, c in enumerate(param[1]):
                    # Coefs: add c
                    coefs[c, :] += param[2][idx, :]
                    # Intercept: add itc value to the correct column
                    intercepts[c] += param[3][idx]

        return [loss_func, classes, coefs / n, intercepts / n]

    return agg
    

# test_LinearRegression.py
# March 2018
#
# This script test the plot_pythia function from the LinearRegression.py script.

import pytest
import pythia_ubc

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model

### Test the plotting function for a continuous feature
def test_plot_pythia():

    diabetes = datasets.load_diabetes()
    X = diabetes.data[:, np.newaxis, 2]

    X_train = X[:-20]
    X_test = X[-20:]

    y_train = diabetes.target[:-20]
    y_test = diabetes.target[-20:]

    # Linear Regression Object
    lm = linear_model.LinearRegression()

    # Train the model using the training sets
    lm.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = lm.predict(X_test)

    # Plot outputs
    scatter_plot = plt.scatter(X_test, y_test,  color='black')
    plt_plot = plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plot_out = plt.show()

    # Expected input
    assert type(lm) == "pythia_ubc.LinearRegression",

    # Expected output
    assert plot_pythia(lm) == type(plot_out), "Plot is NoneType",
    assert type(scatter_plot) == "pythia_ubc.PathCollection",
    assert type(plt_plot) == "list",
    

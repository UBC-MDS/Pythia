# test_LinearRegression.py
# March 2018
#
# This script test the plot_residuals function from the LinearRegression.py script.

import matplotlib
matplotlib.use('Agg')

import sys
import os
sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath("../"))


import pytest
from pythia.LinearRegression import LinearRegression


import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import numpy.random as rand

### Test the plotting function for a continuous feature
def test_plot_residuals():

    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'ones': np.ones(10),
                      'X1': rand.normal(size=10),
                      'X2': rand.normal(size=10),
                      'X3': rand.normal(size=10)})
    y = X.X1 + X.X2 + X.X3 + rand.normal(size=10)

    # Fit a linear regression on the data
    model = LinearRegression(X, y)

    plot = model.plot_residuals()

    # Expected input
    assert isinstance(model, LinearRegression) == True, "The model doesn't have the right type"
    assert isinstance(model.fitted, np.ndarray) == True, "The model's fitted values don't have the right type"
    assert isinstance(model.residuals, np.ndarray) == True, "The model's residuals don't have the right type"

    # Expected output
    #assert isinstance(plot, LinearRegression) == True, "The model doesn't have the right type"
    assert type(plot) == tuple, "The class is the wrong type"
    assert len(plot) == 2, "There are not enough outputs"
    assert type(plot[0]) == type(None), "The output is the wrong type"

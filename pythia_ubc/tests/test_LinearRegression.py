# test_LinearRegression.py
# March 2018
#
# This script test the methods from the LinearRegression.py script.

import pytest
import pythia_ubc

## Packages
import numpy as np
import numpy.random as rand
import pandas as pd

### Test simple linear regression for a continuous feature
def test_LinearRegression1():
    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'X1': rand.normal(size=10)})
    y = X.X1 + rand.normal(size=10)
    
    # Fit a linear regression on the data
    model = LinearRegression(X, y)
    
    # True value of the coefficients
    cov = np.cov(X.X1, y)
    beta = cov[0,1]/cov[0,0]
    alpha = np.mean(y) - beta*np.mean(X.X1)
    fit = alpha + beta*X.X1
    res = y - fit
    
    # Test the type of the output
    assert isinstance(model, pythia_ubc.LinearRegression) == True, "The model doesn't have the right type"
    assert isinstance(model.weights, pd.Series) == True, "The model's weights don't have the right type"
    assert isinstance(model.fitted, pd.Series) == True, "The model's fitted values don't have the right type"
    assert isinstance(model.residuals, pd.Series) == True, "The model's residuals don't have the right type"
    
    # Test the content of the output
    assert model.weights.X1 == beta, "The slope is wrong"
    assert model.weights.intercept == alpha, "The intercept is wrong"
    assert model.fitted == fitted, "The fitted values are wrong"
    assert model.residuals == res, "The residuals are wrong"

### Test multi-linear regression for continuous features
def test_LinearRegression2():
    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'ones': np.ones(10), 
                      'X1': rand.normal(size=10), 
                      'X2': rand.normal(size=10), 
                      'X3': rand.normal(size=10)})
    y = X.X1 + X.X2 + X.X3 + rand.normal(size=10)
    
    # Fit a linear regression on the data
    model = LinearRegression(X, y)
    
    # True value of the coefficients
    X_mat = X.as_matrix()
    beta = np.linalg.inv(np.transpose(X_mat)@X_mat)@np.transpose(X_mat)@np.array(y)
    fit = X_mat@beta
    res = y - fit
    
    # Test the type of the output
    assert isinstance(model, pythia_ubc.LinearRegression) == True, "The model doesn't have the right type"
    assert isinstance(model.weights, pd.Series) == True, "The model's weights don't have the right type"
    assert isinstance(model.fitted, pd.Series) == True, "The model's fitted values don't have the right type"
    assert isinstance(model.residuals, pd.Series) == True, "The model's residuals don't have the right type"
    
    # Test the content of the output
    assert model.weights == beta, "The coefficients are wrong"
    assert model.fitted == fitted, "The fitted values are wrong"
    assert model.residuals == res, "The residuals are wrong"

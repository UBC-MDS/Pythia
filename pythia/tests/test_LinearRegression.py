# test_LinearRegression.py
# March 2018
#
# This script test the methods from the LinearRegression.py script.

import pytest
from pythia_ubc.LinearRegression import LinearRegression

## Packages
import numpy as np
import numpy.random as rand
import pandas as pd
from string import ascii_letters

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
    assert isinstance(model.weights, dict) == True, "The model's weights don't have the right type"
    assert isinstance(model.fitted, np.ndarray) == True, "The model's fitted values don't have the right type"
    assert isinstance(model.residuals, np.ndarray) == True, "The model's residuals don't have the right type"
    
    # Check the length of the output vectors
    assert len(model.weights) == 2
    assert model.fitted.shape == (10,1)
    assert model.residuals.shape == (10,1)

    # Test the content of the output
    assert round(model.weights['X1'], 5) == round(beta, 5), "The slope is wrong"
    assert round(model.weights['intercept'], 5) == round(alpha, 5), "The intercept is wrong"
    assert all(np.around(model.fitted, 5) == np.around(np.array(fit).reshape((10,1)), 5)), "The fitted values are wrong"
    assert all(np.around(model.residuals, 5) == np.around(np.array(res).reshape((10,1)), 5)), "The residuals are wrong"


### Test simple linear regression with duplicate data
def test_LinearRegression2():
    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'X1': rand.normal(size=10)})
    y = X.X1 + rand.normal(size=10)
    X.X1[4] = X.X1[0]
    
    # Fit a linear regression on the data
    model = LinearRegression(X, y)
    
    # True value of the coefficients
    cov = np.cov(X.X1, y)
    beta = cov[0,1]/cov[0,0]
    alpha = np.mean(y) - beta*np.mean(X.X1)
    fit = alpha + beta*X.X1
    res = y - fit
    
    # Test the type of the output
    assert isinstance(model.weights, dict) == True, "The model's weights don't have the right type"
    assert isinstance(model.fitted, np.ndarray) == True, "The model's fitted values don't have the right type"
    assert isinstance(model.residuals, np.ndarray) == True, "The model's residuals don't have the right type"
    
    # Check the length of the output vectors
    assert len(model.weights) == 2
    assert model.fitted.shape == (10,1)
    assert model.residuals.shape == (10,1)

    # Test the content of the output
    assert round(model.weights['X1'], 5) == round(beta, 5), "The slope is wrong"
    assert round(model.weights['intercept'], 5) == round(alpha, 5), "The intercept is wrong"
    assert all(np.around(model.fitted, 5) == np.around(np.array(fit).reshape((10,1)), 5)), "The fitted values are wrong"
    assert all(np.around(model.residuals, 5) == np.around(np.array(res).reshape((10,1)), 5)), "The residuals are wrong"


### Test multi-linear regression for continuous features
### Also check how the function handle non numeric features
def test_LinearRegression3():
    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'X1': rand.normal(size=10), 
                      'X2': rand.normal(size=10), 
                      'X3': rand.normal(size=10),
                      'char': ascii_letters[1:10]})
    y = X.X1 + X.X2 + X.X3 + rand.normal(size=10)
    
    # Fit a linear regression on the data
    model = LinearRegression(X, y)
    
    # True value of the coefficients
    X_mat = X.select_dtypes(include=[np.number], exclude=None)
    X_mat['intercept'] = pd.Series(np.ones(10), index=X_mat.index)
    X_mat = X_mat.as_matrix()
    beta = np.linalg.inv(np.transpose(X_mat)@X_mat)@np.transpose(X_mat)@np.array(y)
    fit = X_mat@beta
    res = y - fit
    
    # Test the type of the output
    assert isinstance(model.weights, dict) == True, "The model's weights don't have the right type"
    assert isinstance(model.fitted, np.ndarray) == True, "The model's fitted values don't have the right type"
    assert isinstance(model.residuals, np.ndarray) == True, "The model's residuals don't have the right type"
    
    # Check the length of the output vectors
    assert len(model.weights) == 4
    assert model.fitted.shape == (10,1)
    assert model.residuals.shape == (10,1)

    # Test the content of the output
    assert all(np.around(model.fitted, 5) == np.around(np.array(fit).reshape((10,1)), 5)), "The fitted values are wrong"
    assert all(np.around(model.residuals, 5) == np.around(np.array(res).reshape((10,1)), 5)), "The residuals are wrong"


### Test multi-linear regression with missing values
def test_LinearRegression4():
    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'X1': rand.normal(size=10), 
                      'X2': rand.normal(size=10), 
                      'X3': rand.normal(size=10)})
    y = X.X1 + X.X2 + X.X3 + rand.normal(size=10)
    
    # Add some missing values
    X.X1[3] = None
    X.X3[5] = None
    
    try:
        LinearRegression(X, y)
    except NameError:
        assert True
    else:
        assert False


### Test error when there is no numeric feature
def test_LinearRegression5():
    # Generate small data to test our function
    X = pd.DataFrame({'char': [ascii_letters[i] for i in range(10)]})
    y = range(10)
    
    try:
        LinearRegression(X, y)
    except NameError:
        assert True
    else:
        assert False
    

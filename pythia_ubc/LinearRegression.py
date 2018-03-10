# LinearRegression.py
# March 2018
#
# This script builds a Linear regression class to analyse data.
# It supports a continuous response and several continuous features.
# The class has a constructor building and fitting the model, and 
# a plotting method for residuals.
#
# Dependencies: 
#
# Usage: 

## Imports
import pandas as pd
import numpy as np

## The LinearRegression class
class LinearRegression:
    """ 
    LinearRegression is a class performing a linear regression on a data frame 
    containing continuous features. 
    
    Its attributes are the coefficients estimates, the fitted values 
    and the residuals from fitting a linear regression of y on X.
    
    Args:
        X: a pandas.dataframe containing continuous variables (including the response)
        y: a pandas.Series of same length containing the response
    
    Attributes: 
        weights: a pandas.Series, the estimated coefficients
        fitted: a pandas.Series, the fitted values
        residuals: a pandas.Series, the residuals
    """ 
    
    def __init__(self, X, y):
        # # Check the type of the features and select the numeric ones
        # cols = (sapply(X, typeof) %in% c('double', 'integer', 'numeric'))
        # X_mat = X %>% select(names(X)[cols])
        # if (sum(cols) > 0) {stop("You need at least one continuous features")}
        
        # Add an intercept column and convert the data frame in a matrix
        n = X.shape[0]
        X_mat = X.copy(deep=True)
        X_mat['intercept'] = pd.Series(np.ones(n), index=X_mat.index)
        names = X_mat.columns
        X_mat = X_mat.as_matrix()
        d = X_mat.shape[1]
        
        # Set hyperparameters
        alpha = 0.001
        n_iter = 1000000
        
        # The gradient of the squared error
        def ols_grad(w):
            return np.dot(np.transpose(X_mat), np.dot(X_mat, weights) - y)
      
        # A norm function for Frobenius
        def norm(x):
            return np.sum(np.abs(x))
    
        # Update the weights using gradient method
        weights = np.zeros(d)
        i = 0
        grad = ols_grad(weights)
        while i < n_iter and norm(grad) > 1e-7:
            grad = ols_grad(weights)
            weights = weights - alpha*grad
            i += 1
        
        temp = {}
        for i in range(len(weights)):
            temp[names[i]] = weights[i]
        self.weights = temp
        
        # Calculate the fitted values
        self.fitted = np.dot(X_mat, weights)
      
        # Calculate the residuals
        self.residuals = y - self.fitted
      
    
    def plot_residuals(self):
        """
        This script makes various diagnostic plots for linear regression analysis.
        It supports a continuous response and several continuous features.

        Args:
            A LinearRegression object containing
                weights: the estimates of the parameters of the linear regression
                fitted: the fitted values
                residuals: the residuals.

        Returns:
            Residuals vs Fitted Plot
            Normal Q-Q Plot
            Fitted vs True Value Plot(s)
        """
        pass
    

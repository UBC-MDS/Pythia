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
        pass
    
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
    
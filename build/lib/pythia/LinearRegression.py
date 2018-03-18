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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import numpy.random as random

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
        # Check the type of the features and select the numeric ones
        X_mat = X.select_dtypes(include=[np.number], exclude=None)
        if X_mat.shape[1] == 0:
            raise NameError("You need at least one continuous features")
        
        try:
            for var in X_mat.columns:
                assert np.all(X_mat[[var]].notnull())
        except AssertionError:
            raise NameError("Some of your numeric features contain missing values. Please deal with them (remove, impute...) before using this function.")
        else:
            # Add an intercept column and convert the data frame in a matrix
            n = X_mat.shape[0]
            X_mat['intercept'] = pd.Series(np.ones(n), index=X_mat.index)
            names = X_mat.columns
            X_mat = X_mat.as_matrix()
            d = X_mat.shape[1]
            y = np.array(y).reshape((10,1))
            
            # Set hyperparameters
            alpha = 0.001
            n_iter = 1000000
            
            # The gradient of the squared error
            def ols_grad(w):
                return np.dot(np.transpose(X_mat), np.dot(X_mat, w) - y)
            
            # A norm function for Frobenius
            def norm(x):
                return np.sum(np.abs(x))
        
            # Update the weights using gradient method
            weights = np.zeros(d).reshape((d,1))
            i = 0
            grad = ols_grad(weights)
            while i < n_iter and norm(grad) > 1e-7:
                grad = ols_grad(weights)
                weights = weights - alpha*grad
                i += 1
            
            temp = {}
            for i in range(len(weights)):
                temp[names[i]] = weights[i,0]
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
        assert len(self.residuals) > 0, "There are no residuals"	
        assert len(self.fitted) > 0, "There are no fitted values"	
        assert len(self.residuals) == len(self.fitted), "The number of residuals and fitted values do not match"	
        	
        # Get fitted values and residuals	
        residuals = self.residuals	
        fitted = self.fitted	
        
        residuals = residuals.flatten()
        fitted = fitted.flatten()
        	
        # Fitted vs Residuals	
        plt.figure(figsize=(10,6))	
        plt.scatter(fitted, residuals,  color='grey')	
        plt.axhline(y = 0, linewidth = 1, color = 'red')	
        plt.xlabel('Fitted Values')	
        plt.ylabel('Residuals')	
        plt.title('Residuals vs. Fitted Values')	
        resfit = plt.show()	
        	
        # Normal QQ Plot	
        res = np.asarray(residuals)	
        res.sort()	
        	
        # Generate normal distribution	
        ndist = random.normal(loc = 0, scale = 1, size = len(res))	
        ndist.sort()	
	
        # Fit Normal Trendline.  	
        fit = np.polyfit(ndist, res, 1)	
        fit = fit.tolist()
        func = np.poly1d(fit)	
        trendline_y = func(ndist)	
	
        plt.figure(figsize=(10,6)) 	
        plt.scatter(ndist, res, color = 'grey')	
        plt.plot(ndist, trendline_y, color = 'red')	
        plt.title("Normal QQ Plot")	
        plt.xlabel("Theoretical quantiles")	
        plt.ylabel("Expreimental quantiles")	
        qqplot = plt.show()	
        	
        return (resfit,qqplot)
    

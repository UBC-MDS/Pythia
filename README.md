# Pythia
This Python package is a Linear Regression tool.

## Group Members:
Maud Boucherit:  [Personal Github](https://github.com/MaudBoucherit)  
Ted Haley: [Personal Github](https://github.com/TedHaley)  
Cem Sinan Ozturk:  [Personal Github](https://github.com/cemsinano)  

## Description:
This package will take a dataset with a continuous reponse variable and various explanatory variables (either continuous, discrete, or categorical), and provide the user with several functions to bulid a linear regression model. 

## Functions:  
#### Summary of Data: (input: X, y)    
This function will return a table containing various statistics from the provided dataset. These statistics include the mean, variance, quantile for continuous variables, mode, number of levels (for categorical data), etc. 

#### Model Fit: (input: X, y)  
This function will return the model object containing the coefficients for the linear model, corresponding p-value, and confidence interval.

#### Residual Plot: (input: Model Object) 
This function takes the calculated residuals from the model object and will display both the residuals by fitted values, and the QQ plot for the residuals.

## Ecosystem:
Our package runs similarly to the Scikit Learn LinearRegression class that already exist in Python.

The statistics summary from the Pythia package is similar to the pandas describe function. The Pythia summary statistics function will return a pandas dataframe containing all relevant statistics dependening on the variable type (continuous, discrete, or categorical).

The model fit function works similarly to LinearRegression.fit function from the Scikit Learn package. Both functions return a model object, and the coefficients can be extracted from the model object. 

The residual plot function is a method for the model class that returns two plots specific to the residual. There are functions that exist to plot both the residual-fitted plot and the QQ-plot, however the function from the Karl package combines these two plots in one function, displaying both plots in tandem. 


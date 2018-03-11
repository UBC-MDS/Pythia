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

This function will return a table containing various statistics from the provided dataset. These statistics include the mean, variance, minimum, maximum and quantile (25, 50 and 75) values for continuous variables.

#### Model Fit: (input: X, y)  

This function will return the model object containing the coefficients for the linear model, corresponding p-value, and confidence interval.

#### Residual Plot: (input: Model Object) 
This function takes the calculated residuals from the model object and will display both the residuals by fitted values, and the QQ plot for the residuals.

## Ecosystem:
Our package runs similarly to the Scikit Learn LinearRegression class that already exist in Python.

The statistics summary from the Pythia package is similar to the pandas describe function. The Pythia summary statistics function will return a pandas dataframe containing all relevant statistics dependening on the variable type (continuous, discrete, or categorical).

The model fit function works similarly to LinearRegression.fit function from the Scikit Learn package. Both functions return a model object, and the coefficients can be extracted from the model object. 

The residual plot function is a method for the model class that returns two plots specific to the residual. There are functions that exist to plot both the residual-fitted plot and the QQ-plot, however the function from the Karl package combines these two plots in one function, displaying both plots in tandem. 

## Installation:
To install Pythia, follow these instructions:  
1. Input the following into the Terminal: `pip install git+https://github.com/UBC-MDS/Pythia.git`  
2. You're ready to start using `Pythia`!

## Usage:
#### Summary of Data: `EDA(X, y)`   
This function will return a table containing various statistics from the provided dataset. These statistics include the mean, variance, minimum, maximum and quantile (25, 50 and 75) values for continuous variables.

Arguments:

   - X: a pandas.dataframe containing continuous variables 
   - y: a pandas.Series of same length containing the response

Values: a dataframe containing 

  - mean: the mean for response (y) and features (X)
  - variance: the variance for response (y) and features (X)
  - quantiles: the 25-50-75 quantiles for response (y) and features (X)
  - min: the minimum value for response (y) and features (X)
  - max: the maximum value for response (y) and features (X)
  
Usage: 

```
import pythia_ubc
X = pd.DataFrame({'ones': np.ones(10),
                  'X1': rand.normal(size=10),
                  'X2': rand.normal(size=10),
                  'X3': rand.normal(size=10)})
y = pd.DataFrame({X.X1 + X.X2 + X.X3 + rand.normal(size=10)})
EDA(X,y)
```
Expected Output: 

|       | mean  | variance | min | quantile25 | quantile50 | quantile75 | max |
|-------|-------|----------|-----|------------|------------|------------|-----|
|   y   |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |
|   X1  |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |
|   X2  |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |
|   X3  |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |

#### Model Fit: `LinearRegression(X, y)` 

 This function returns a method object containing the weights, fitted values, and residuals from fitting a linear regression of y on X.
 
Arguments:
 
   - X: a pandas.dataframe containing continuous variables 
   - y: a pandas.Series of same length containing the response

Values: a class method containing:

  - weights: a pandas.Series, the estimated coefficients
  - fitted: a pandas.Series, the fitted values
  - residuals: a pandas.Series, the residuals

Usage:

```
import pythia_ubc
X = pd.DataFrame({'X1': np.random.normal(size=10), 
                  'X2': np.random.normal(size=10),
                  'X3': np.random.normal(size=10)})
y = X.X1 + X.X2 + X.X3 + np.random.normal(size=10)
LinearRegression(X, y)
```

#### Residual Plot: `plot_residuals((input: Model Object))` 

This function is used to plot the linear model object from the LinearRegression class. The linear model object is a method that includes weights, fitted values, and residuals. This function will return 2 types of plots, which include:

  - Residuals vs Fitted Plot
  - Normal Q-Q Plot

Arguments:

  - lm object: a list of lists containing:
  	- fitted: the fitted values
  	- residuals: the residuals.

Value:

  - Residuals vs Fitted Plot
  - Normal Q-Q Plot
  
Usage:

```
import pythia_ubc
lm = LinearRegression(X, y)
plot_residuals(lm)
```

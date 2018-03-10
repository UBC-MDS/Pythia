# eda.py
# March 2018
#
# This script contains a function producing an EDA (explanatory data analysis) summary of the data.
#
# The dataset with a continuous response variable and
# various continuous explanatory variables
#
#
# Dependencies:
#   - numpy
#   - pandas
#   - numpy
#
# Usage:
#
# import pythia_ubc
#
# pythia_ubc.eda(X,y)
#
# Imports

import pytest
import numpy as np
import pandas as pd
#
# eda (explanatory data analysis) summary function:
def eda(X):
    """
    eda(X,y)

    EDA
    This function returns a dataframe containing mean,
    variance, min, max and quantiles for each variables in the dataset

    Args:
        X: a dataframe containing continuous features
        y: a numeric vector of same length containing the response

    Attributes: a dataframe containing
        mean: the mean for response (y) and features (X)
        variance: the variance for response (y) and features (X)
        quantiles: the 25-50-75 quantiles for response (y) and features (X)
        min: the minimum value for response (y) and features (X)
        max: the maximum value for response (y) and features (X)

    Example Usage:

        # import required packages

        import pythia_ubc
        import numpy as np
        import numpy.random as rand
        import pandas as pd

        # set a random seed:

        rand.seed(4)

        # create a mock dataframe to exemplify the usage and results

        X = pd.DataFrame({'ones': np.ones(10),
                          'X1': rand.normal(size=10),
                          'X2': rand.normal(size=10),
                          'X3': rand.normal(size=10)})
        y = X.X1 + X.X2 + X.X3 + rand.normal(size=10)

        # get EDA summary for the data:

        pythia_ubc.eda(X, y)

        Results will be presented in a dataframe in the following form:

        |       | mean  | variance | min | quantile25 | quantile50 | quantile75 | max |
        |-------|-------|----------|-----|------------|------------|------------|-----|
        |   y   |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |
        |   X1  |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |
        |   X2  |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |
        |   X3  |   ... |  ...     | ... |    ...     |     ...    |     ...    | ... |
    """
    allData = pd.concat([y.reset_index(drop=True), X], axis=1)
    allData.rename({0: 'y'})

    summary = pd.DataFrame({
    'mean': allData.mean(axis=0),
    'var': allData.var(axis=0),
    'min': allData.min(axis=0),
    'quantile25': allData.quantile(.25),
    'quantile50': allData.quantile(.50),
    'quantile75': allData.quantile(.75),
    'max': allData.max(axis=0)
    })

    return summary

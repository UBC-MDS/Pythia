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
def eda(X,y):
    """
    eda(X,y)

    EDA
    This function returns a dataframe containing mean,
    variance, min, max and quantiles for each variables in the dataset

    Args:
        X: a pandas.dataframe containing continuous variables (including the response)
        y: a pandas.dataframe vector (a pandas.Series) of same length containing the response

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

        X = pd.DataFrame({'X1': rand.normal(size=10),
                          'X2': rand.normal(size=10),
                          'X3': rand.normal(size=10)})
        y = pd.DataFrame({X.X1 + X.X2 + X.X3 + rand.normal(size=10)})

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

    assert len(X) > 0, "There are no values in features"
    assert len(y) > 0, "There are no values in response"
    assert len(X) == len(y), "Length oh response and features does not match"

    # Test the type of the input
    assert isinstance(X, pd.DataFrame) == True, "The features(X) doesn't have the right type"
    assert isinstance(y, pd.DataFrame) == True, "The response(y) don't have the right type"

    # Check the type of the features and select the numeric ones to summarize
    X = X.select_dtypes(include=[np.number], exclude=None)
    if X.shape[1] == 0:
        raise NameError("You do not have any continuous features to summarize")

    allData = pd.concat([y.reset_index(drop=True), X], axis=1)
    allData.rename({0: 'y'})

    summary = pd.DataFrame({
    'mean': allData.mean(axis=0),
    'variance': allData.var(axis=0),
    'min': allData.min(axis=0),
    'quantile25': allData.quantile(.25),
    'quantile50': allData.quantile(.50),
    'quantile75': allData.quantile(.75),
    'max': allData.max(axis=0)
    })

    return summary

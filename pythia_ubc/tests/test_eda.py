# test_eda.py
# March 2018
#
# This script test the summary function from eda.py.

import pytest
import pythia_ubc

## Packages
import numpy as np
import numpy.random as rand
import pandas as pd

### Test eda(explanatory data analysis) function for continuous features
def test_eda():
    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'X1': rand.normal(size=10),
                      'X2': rand.normal(size=10),
                      'X3': rand.normal(size=10)})
    y = pd.DataFrame({'y':X.X1 + X.X2 + X.X3 + rand.normal(size=10)})

    # get EDA summary for the data
    summary = eda(X, y)

    # True value of the mean, variance and quantiles:
    # `mean`, `var` and `percentile` are numpy functions that I will compare results.

    # mean values:
    X1_mean = np.mean(X.X1)
    X2_mean = np.mean(X.X2)
    X3_mean = np.mean(X.X3)
    y_mean = np.mean(y)

    # variance values:
    X1_var = np.var(X.X1, ddof=1)
    X2_var = np.var(X.X2, ddof=1)
    X3_var = np.var(X.X3, ddof=1)
    y_var = np.var(y, ddof=1)

    ## quantiles

    # 0% (minimum) :
    X1_min = np.percentile(X.X1, 0)
    X2_min = np.percentile(X.X2, 0)
    X3_min = np.percentile(X.X3, 0)
    y_min = np.percentile(y, 0)

    # 25%
    X1_quantile25 = np.percentile(X.X1, 25)
    X2_quantile25 = np.percentile(X.X2, 25)
    X3_quantile25 = np.percentile(X.X3, 25)
    y_quantile25 = np.percentile(y, 25)

    # 50%
    X1_quantile50 = np.percentile(X.X1, 50)
    X2_quantile50 = np.percentile(X.X2, 50)
    X3_quantile50 = np.percentile(X.X3, 50)
    y_quantile50 = np.percentile(y, 50)

    # 75%
    X1_quantile75 = np.percentile(X.X1, 75)
    X2_quantile75 = np.percentile(X.X2, 75)
    X3_quantile75 = np.percentile(X.X3, 75)
    y_quantile75 = np.percentile(y, 75)

    # 100% (maximum) :
    X1_max = np.percentile(X.X1, 100)
    X2_max = np.percentile(X.X2, 100)
    X3_max = np.percentile(X.X3, 100)
    y_max = np.percentile(y, 100)

    # Test the type of the input
    assert isinstance(X, pd.DataFrame) == True, "The features(X) doesn't have the right type"
    assert isinstance(y, pd.DataFrame) == True, "The response(y) don't have the right type"

    # Test the type of the output
    assert isinstance(summary, pd.DataFrame) == True, "The model doesn't have the right type"
    assert isinstance(summary, pd.DataFrame) == True, "The model don't have the right type"
    assert list(summary.columns) == ['max', 'mean', 'min', 'quantile25', 'quantile50', 'quantile75', 'variance'], "The summary dataframe's columnnames are not as expected"
    assert isinstance(summary.mean(), pd.Series) == True, "The model's mean don't have the right type"
    assert isinstance(summary.variance, pd.Series) == True, "The model's variance don't have the right type"
    assert isinstance(summary.min(), pd.Series) == True, "The model's maximum don't have the right type"
    assert isinstance(summary.quantile25, pd.Series) == True, "The model's 25th quantile don't have the right type"
    assert isinstance(summary.quantile50, pd.Series) == True, "The model's 50th quantile don't have the right type"
    assert isinstance(summary.quantile75, pd.Series) == True, "The model's 75th quantile don't have the right type"
    assert isinstance(summary.max(), pd.Series) == True, "The model's maximum don't have the right type"

    # Test the content of the output

    # mean values:
    assert summary['mean'][0] == y_mean[0], "Mean of response(y) is wrong"
    assert summary['mean'][1] == X1_mean, "Mean of explanatory variable X1 is wrong"
    assert summary['mean'][2] == X2_mean, "Mean of explanatory variable X2 is wrong"
    assert summary['mean'][3] == X3_mean, "Mean of explanatory variable X3 is wrong"

    # variance values:
    assert summary.variance[0] == y_var[0], "variance of response(y) is wrong"
    assert summary.variance[1] == X1_var, "variance of explanatory variable X1 is wrong"
    assert summary.variance[2] == X2_var, "variance of explanatory variable X2 is wrong"
    assert summary.variance[3] == X3_var, "variance of explanatory variable X3 is wrong"

    # min (Oth percentile) values:
    assert summary['min'][0] == y_min, "minimum of response(y) is wrong"
    assert summary['min'][1] == X1_min, "minimum of explanatory variable X1 is wrong"
    assert summary['min'][2] == X2_min, "minimum of explanatory variable X2 is wrong"
    assert summary['min'][3] == X3_min, "minimum of explanatory variable X3 is wrong"

    # 25th percentile values:
    assert summary.quantile25[0] == y_quantile25, "25th quantile of response(y) is wrong"
    assert summary.quantile25[1] == X1_quantile25, "25th quantile of explanatory variable X1 is wrong"
    assert summary.quantile25[2] == X2_quantile25, "25th quantile of explanatory variable X2 is wrong"
    assert summary.quantile25[3] == X3_quantile25, "25th quantile of explanatory variable X3 is wrong"

    # 50th percentile values:
    assert summary.quantile50[0] == y_quantile50, "50th quantile of response(y) is wrong"
    assert summary.quantile50[1] == X1_quantile50, "50th quantile of explanatory variable X1 is wrong"
    assert summary.quantile50[2] == X2_quantile50, "50th quantile of explanatory variable X2 is wrong"
    assert summary.quantile50[3] == X3_quantile50, "50th quantile of explanatory variable X3 is wrong"

    # 75th percentile values:
    assert summary.quantile75[0] == y_quantile75, "75th quantile of response(y) is wrong"
    assert summary.quantile75[1] == X1_quantile75, "75th quantile of explanatory variable X1 is wrong"
    assert summary.quantile75[2] == X2_quantile75, "75th quantile of explanatory variable X2 is wrong"
    assert summary.quantile75[3] == X3_quantile75, "75th quantile of explanatory variable X3 is wrong"

    # max (10Oth percentile) values:
    assert summary['max'][0] == y_max, "maximum of response(y) is wrong"
    assert summary['max'][1] == X1_max, "maximum of explanatory variable X1 is wrong"
    assert summary['max'][2] == X2_max, "maximum of explanatory variable X2 is wrong"
    assert summary['max'][3] == X3_max, "maximum of explanatory variable X3 is wrong"


def test_eda2():
    # Data for the error case:
    X = "aString"
    y = pd.DataFrame({'y': rand.normal(size=10))

    try:
        eda(X, y)
    except NameError:
        assert True
    else:
        assert False

def test_eda3():
    # Data for the error case:
    X = pd.DataFrame({'X1': rand.normal(size=10)
    y = "notDataframe"

    try:
        eda(X, y)
    except NameError:
        assert True
    else:
        assert False

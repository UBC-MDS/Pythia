# Integration Test
# pip install git+https://github.com/UBC-MDS/Pythia.git

# Import Pythia Package
from pythia.LinearRegression import LinearRegression
from pythia.eda import eda

# Import random
import numpy.random as rand
import pandas as pd

def test_integration():

    # Generate small data to test our function
    rand.seed(4)
    X = pd.DataFrame({'X1': rand.normal(size=10),
                      'X2': rand.normal(size=10),
                      'X3': rand.normal(size=10)})
    y = pd.DataFrame({'y':X.X1 + X.X2 + X.X3 + rand.normal(size=10)})

    # get EDA summary for the data
    summary = eda(X, y)

    # LinearRegression Function is a class.
    # Fit a linear regression on the data
    model = LinearRegression(X, y)
    
    # Plot residuals is dependent on LinearRegression Function.
    plot = model.plot_residuals()

    # Expected output
    #assert isinstance(plot, LinearRegression) == True, "The model doesn't have the right type"
    assert type(plot) == tuple, "The class is the wrong type"
    assert len(plot) == 2, "There are not enough outputs"
    assert type(plot[0]) == type(None), "The output is the wrong type"

# Integration Test
# pip install git+https://github.com/UBC-MDS/Pythia.git

# Import Pythia Package
from pythia.LinearRegression import LinearRegression
from pythia.eda import eda

X = pd.DataFrame({'ones': np.ones(10),
                  'X1': rand.normal(size=10),
                  'X2': rand.normal(size=10),
                  'X3': rand.normal(size=10)})
y = pd.DataFrame({X.X1 + X.X2 + X.X3 + rand.normal(size=10)})

# Summary function is independent of LinearRegression function
summary = EDA(X,y)
print(summary)

# LinearRegression Function is a class.
lm = LinearRegression(X, y)
print(lm)

# Plot residuals is dependent on LinearRegression Function.
LinearRegression.plot_residuals(lm)

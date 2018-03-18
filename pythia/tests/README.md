# Package Pythia - Tests

This package contains three functions. Unit tests were produced for each of them.

## test `EDA(X, y)`
This function first tests the type/class of the input, in this case `X` a data frame containing three numeric variables.   
Then, it tests the type/class and column names of the output, a data frame containing statistics.   
Finally, it compares the content of the data frame, i.e. several statistics for each variables from the input, with the same statistic computed "by hand".

## test `LinearRegression(X, y)`
This function tests the output type and its content: the elements' type, shape and values, for several `X` input:
- one continuous feature,
- one continuous feature with duplicated observations,
- three continuous features and one character feature.

It also tests if an error is return when there is no continuous feature.

## test `plot_pythia(lm)`
This function first tests the input of the `plot_karl` function, to be sure that what is used is what should be used.   
Then, it tests the type and length of the output.

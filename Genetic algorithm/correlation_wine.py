import pandas as pd

# URL for the red wine dataset
red_wine_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

# Load the dataset
red_wine = pd.read_csv(red_wine_url, sep=';')

# Compute the correlation matrix
correlation_matrix = red_wine.corr()

# Extracting the correlations with the 'quality' column
quality_correlations = correlation_matrix['quality'].sort_values(ascending=False)
print(quality_correlations)
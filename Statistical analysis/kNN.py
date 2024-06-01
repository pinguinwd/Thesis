#%%
from MNUECA_Lyapunov import *
import numpy as np
import pandas as pd
import os
from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

class type:
    name = 'Generate'
    firstrules = [i for i in range(256)]
    secondrules = [255]

def generate_MNUECA_Lyapunov_data():
    STEPS = 100
    SIZE = 101
    file_path = r'D:/PythonProjects/Thesis-data/smallCA/Data/'
    lyapnov_matrix, kolmogorov_matrix, shannon_matrix = generate_matrices(file_path, STEPS, SIZE, type)
    save_data(lyapnov_matrix, 'lyapnov_matrix')
    save_data(kolmogorov_matrix, 'kolmogorov_matrix')
    save_data(shannon_matrix, 'shannon_matrix')

def create_array_for_DBSCAN(path):
    # List all CSV files in the specified directory
    files = [f for f in os.listdir(path) if f.endswith('.csv')]
    files.sort()  # Sorting ensures consistent order

    # Ensure there are CSV files to process
    if not files:
        raise ValueError("No CSV files found in the directory.")

    # Read the first file to determine the dimensions and initialize the data array
    sample_data = pd.read_csv(os.path.join(path, files[0]))
    rows, cols = sample_data.shape

    # Create an array to hold data from all files
    # Shape will be (number of coordinates, number of files)
    all_data = np.empty((rows * cols, len(files)), dtype=sample_data.dtypes[0])

    # Load data from each file
    for file_index, filename in enumerate(files):
        full_path = os.path.join(path, filename)
        data = pd.read_csv(full_path).values
        if data.shape != (rows, cols):
            raise ValueError(f"All CSV files must have the same dimensions. Problem with file: {filename}")

        # Flatten the data and stack it in 'all_data'
        all_data[:, file_index] = data.flatten()

    return all_data

def create_and_show_clusters(training_data, path):
    # Define the model
    dbscan_model = DBSCAN(eps=0.25, min_samples=10)

    # Train the model
    dbscan_model.fit(training_data)

    # Assign each data point to a cluster
    dbscan_result = dbscan_model.labels_

    # Get all of the unique clusters
    dbscan_clusters = np.unique(dbscan_result)

    # Plot the DBSCAN clusters
    pyplot.figure(figsize=(8, 6))  # Set the figure size
    for dbscan_cluster in dbscan_clusters:
        # Get data points that fall in this cluster
        index = np.where(dbscan_result == dbscan_cluster)
        # Make the plot
        pyplot.scatter(training_data[index, 0], training_data[index, 1], label=f'Cluster {dbscan_cluster}')

    # Labeling the plot
    pyplot.title('DBSCAN Clustering')
    pyplot.xlabel('Feature 1')
    pyplot.ylabel('Feature 2')

    # Save the DBSCAN plot
    im_path = path + '\\clustering.png'
    pyplot.savefig(im_path)  # Save the figure to the specified path
    pyplot.close()  # Close the figure after saving to free up memory

    print(f'Cluster plot saved to {im_path}')


if __name__ == '__main__':
    #generate_MNUECA_Lyapunov_data()
    path = 'D:\\PythonProjects\\Thesis-data\\smallCA\\Analysis\\Plotting\\Without langton'
    all_data = create_array_for_DBSCAN(path)
    create_and_show_clusters(all_data, path)
# %%

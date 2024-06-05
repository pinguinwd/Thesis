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

#%%
path = 'D:\\PythonProjects\\Thesis\\Statistical analysis\\Statistical testing'
data_path = path + '\\information.csv'
training_data = pd.read_csv(data_path, index_col=['Rule'])
#%%

def create_and_show_clusters(training_data, path, features, task = 'show'):
    training_array = np.array(training_data)
    
    # Define the model
    dbscan_model = DBSCAN(eps=0.25, min_samples=10)

    # Train the model
    dbscan_model.fit(training_array)

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
        pyplot.scatter(training_data[features[0]], training_data[features[1]], label=f'Cluster {dbscan_cluster}')

    # Labeling the plot
    pyplot.title('DBSCAN Clustering')
    pyplot.xlabel(features[0])
    pyplot.ylabel(features[1])

    # Save the DBSCAN plot
    im_path = path + '\\clustering.png'
    if task == 'show':
        pyplot.show()
    if task == 'save':
        pyplot.savefig(im_path)  # Save the figure to the specified path
        pyplot.close()  # Close the figure after saving to free up memory

    print(f'Cluster plot saved to {im_path}')

create_and_show_clusters(training_data, path, ['Average', 'Standard deviation'], 'show')
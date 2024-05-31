#%%
import numpy as np
import re
import math
import os
import pandas as pd
import zlib
import matplotlib.pyplot as plt
from icecream import ic

#%%
def parse_binary_arrays(file_path, STEPS, SIZE):
    # List to store the numpy arrays
    array_list = []
    
    # Open the text file for reading
    with open(file_path, 'r') as file:
        # Read the entire content of the file
        content = file.read()
        
        # Find all the patterns that match the [numbers] format using regular expression
        pattern = r'\[(.*?)\]'
        matches = re.findall(pattern, content)
        
        # Process each match
        for match in matches:
            # Split the string by comma and convert each to int
            numbers = list(map(int, match.split(',')))
            
            # Check if the numbers list is the expected length of 10100
            if len(numbers) != 10100:
                raise ValueError("Each list of numbers must contain exactly 10100 elements.")
            
            # Convert the list to a numpy array and reshape to 100x101
            array = np.array(numbers).reshape(STEPS, SIZE)
            
            # Append the numpy array to the list
            array_list.append(array)
    
    # Return the list of numpy arrays
    return array_list

def calculate_lyapunov(defect_cone):
    # Calculate the Hamming distance for each row (time step) in the defect cone
    hamming_distances = np.sum(defect_cone, axis=1)

    # Compute the logarithm of distances to estimate the Lyapunov exponent
    log_distances = np.log(hamming_distances + 1e-10)  # Adding a small constant to avoid log(0)
    time_steps = np.arange(len(hamming_distances))
    
    # Linear regression to estimate the slope (Lyapunov exponent)
    coefficients = np.polyfit(time_steps, log_distances, 1)
    return coefficients[0]

def calculate_shannon(defect_cone):
    last_row = defect_cone[-1]  # Get the last row of the defect cone
    n_cells = last_row.size  # Total number of cells in the row

    # Count the number of 0s and 1s
    count_0 = np.count_nonzero(last_row == 0)
    count_1 = n_cells - count_0  # Since only two possible states exist

    # Calculate the probabilities
    p0 = count_0 / n_cells
    p1 = count_1 / n_cells

    # Calculate the Shannon entropy
    entropy = 0
    if p0 > 0:
        entropy -= p0 * np.log2(p0)
    if p1 > 0:
        entropy -= p1 * np.log2(p1)

    return entropy

def calculate_kolmogorov(defect_cone):
    # Convert the numpy array to bytes
    data_bytes = defect_cone.tobytes()
    # Compress the byte data
    compressed_data = zlib.compress(data_bytes)
    # Return the length of the compressed data as the complexity estimate
    return len(compressed_data)

def evaluate_one_textfile(path, STEPS, SIZE):
    array_list = parse_binary_arrays(path, STEPS, SIZE)

    lyapunov_coefs = []
    kolmogorov_coefs = []
    shannon_coefs = []

    for array in array_list:
        lyapunov = calculate_lyapunov(array)
        lyapunov_coefs.append(lyapunov)

        kolmogorov = calculate_kolmogorov(array)
        kolmogorov_coefs.append(kolmogorov)

        shannon = calculate_shannon(array)
        shannon_coefs.append(shannon)

    return np.average(lyapunov_coefs), np.average(kolmogorov_coefs), np.average(shannon_coefs)

def generate_matrices(path, STEPS, SIZE):
    #lypanov
    lyapnov_matrix = np.zeros((256,256))

    #Kolmogorov Complexity
    kolmogorov_matrix = np.zeros((256,256))

    #Shannon Entropy
    shannon_matrix = np.zeros((256,256))


    # Walk through all directories and files in the given path
    for dirpath, dirnames, filenames in os.walk(path):
        for filename in filenames:
            # Check if the filename matches the required pattern
            match = re.match(r'Sim_firstrule_(\d+)_secondrule_(\d+)\.txt', filename)
            if match:
                x, y = int(match.group(1)), int(match.group(2))
                full_path = os.path.join(dirpath, filename)
                # Evaluate the text file and store the result in a dictionary with (x, y) as key
                ic(full_path)
                lyapnov_matrix[x,y], kolmogorov_matrix[x,y], shannon_matrix[x,y] = evaluate_one_textfile(full_path, STEPS, SIZE)
    
    return lyapnov_matrix, kolmogorov_matrix, shannon_matrix

#%%

def save_data(data, name):
    df = pd.DataFrame(data)
    df.to_csv('D:\\PythonProjects\\Thesis-data\\smallCA\\Analysis\\'+ str(name) + '.csv', index=False)
    

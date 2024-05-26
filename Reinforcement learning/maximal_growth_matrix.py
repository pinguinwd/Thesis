#%%
import pandas as pd
import numpy as np
from generate_data_for_RL import *
from tqdm import tqdm
#%%
STEPS = 13
CA_SIZE = 25
train_split = 0.5

red_wine = pd.read_csv('D:\\PythonProjects\\Thesis\\red_wine.csv')
# %%

def generate_matrix(inputs, outputs):
    # Create a DataFrame with rows 0-15 and columns 3-8
    df = pd.DataFrame(0.0, index=np.arange(16), columns=np.arange(3, 9))
    
    # Iterate over input-output pairs
    for input, output in zip(inputs, outputs):
        # Ensure the input is a numpy array
        input_array = np.array(input)
        output_array = np.array(output)

        # Convert input binary array to a decimal number
        input_number = int(''.join(input_array.astype(str)), 2)
        
        # Iterate over numbers 3 through 8 and calculate the fitness
        for num in range(3, 9):
            # Convert number to binary, pad it to 4 bits
            compare_bin = np.array(list(f"{num:04b}"), dtype=int)
            
            # Calculate the hamming distance
            hamming_distance = np.sum(output_array != compare_bin)
            
            # Calculate the fitness
            fitness = 1 / (1 + hamming_distance)
            
            # Add the fitness value to the appropriate cell in the DataFrame
            df.at[input_number, num] += fitness
    
    return df

def evaluate_matrix(matrix, inputs, outputs):
    result_dict = {}
    for index, row in matrix.iterrows():
        # Find the column index with the highest fitness score in the row
        max_fitness_col = row.idxmax()
        
        # Convert index (decimal number) to a four-bit binary list
        index_bin = list(map(int, list(f"{index:04b}")))
        
        # Convert max_fitness_col (decimal number, which needs correction from typo) to a four-bit binary list
        max_fitness_col_bin = list(map(int, list(f"{max_fitness_col:04b}")))
        
        # Add to the result dictionary
        result_dict[tuple(index_bin)] = tuple(max_fitness_col_bin)
    
    # Evaluate the result_dict against actual inputs and outputs
    total_fitness = 0
    count = 0

    for input, output in zip(inputs, outputs):
        # Ensure input is a numpy array and convert to tuple
        input_tuple = tuple(map(int, input))

        # Predict using result_dict
        predicted_output = result_dict.get(input_tuple)
        
        # Convert output to tuple for comparison
        output_tuple = tuple(map(int, output))

        # Calculate the Hamming distance if prediction was possible
        if predicted_output:
            hamming_distance = sum(1 for x, y in zip(predicted_output, output_tuple) if x != y)
            fitness = 1 / (1 + hamming_distance)
            total_fitness += fitness
            count += 1
    
    # Calculate average fitness
    final_fitness = total_fitness / count if count > 0 else 0

    return result_dict, final_fitness


# %%
fitness = []
for i in tqdm(range(1000), desc="Best possible for different settings"):
    (input_train, output_train), (input_control, output_control) = create_input_output(red_wine, train_split=0.8)
    matrix = generate_matrix(input_train, output_train)
    best_option, final_fitness = evaluate_matrix(matrix, input_train, output_train)
    fitness.append(final_fitness)
print(fitness)
pd.DataFrame([fitness]).to_csv('D:\\PythonProjects\\Thesis\\Reinforcement learning\\fitness.csv', index=False, header=False)
# %%

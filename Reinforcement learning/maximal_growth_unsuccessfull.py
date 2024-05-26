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

def check_one_option(prediction, input, output):
    # Convert input and output lists to NumPy arrays if they aren't already
    if not isinstance(input, np.ndarray):
        input = np.array(input)
    if not isinstance(output, np.ndarray):
        output = np.array(output)
    
    total_fitness = 0
    num_inputs = len(input)

    # Iterate over each input-output pair
    for i in range(num_inputs):
        inp_tuple = tuple(input[i])
        predicted_output = np.array(prediction[inp_tuple])
        
        # Calculate the distance using NumPy's vectorized operations
        distance = np.sum(predicted_output != output[i])
        
        # Update total fitness with the new calculated fitness
        total_fitness += 1 / (1 + distance)
    
    # Return the average fitness
    return total_fitness / num_inputs if num_inputs else 0

def hex_to_binary_dict(hex_string):
    # Ensure the hex string has exactly 16 characters
    assert len(hex_string) == 16, "The hexadecimal string must have exactly 16 digits."

    # Initialize the dictionary with all possible 4-bit binary keys
    result_dict = {tuple(map(int, format(x, '04b'))): None for x in range(16)}

    # Populate the dictionary with values from the hexadecimal string
    for i, char in enumerate(hex_string):
        # Convert the hexadecimal character to an integer, then format it as a binary with 4 digits
        hex_to_bin = tuple(np.array(list(f"{int(char, 16):04b}"), dtype=int))

        # Convert decimal number to a binary numpy array with 4 digits
        dec_to_bin = tuple(np.array(list(f"{i:04b}"), dtype=int))

        # Assign the binary tuple both as a key and a value to the dictionary
        result_dict[dec_to_bin] = hex_to_bin

    # Ensure all entries have a valid binary tuple value; if not, set them to their keys (self-mapping)
    for key in result_dict:
        if result_dict[key] is None:
            result_dict[key] = key

    return result_dict

def transform_decimal_to_base6(number):
    # Set a fixed number of digits to 16
    num_digits = 16

    # Initialize an empty array for digits
    digits = np.zeros(num_digits, dtype=int)
    
    # Fill the array with base-6 digits
    for i in range(num_digits):
        if number == 0:
            break
        digits[num_digits - 1 - i] = number % 6
        number //= 6

    # Add 3 to each digit to make digits range from 3 to 8
    # Since base 6 normally has digits 0-5, adding 3 makes them 3-8 directly without modulo
    transformed_digits = digits + 3

    # Convert array of digits to a string
    result = ''.join(transformed_digits.astype(str))
    return result



def iterate_over_possible_dict(input, output, begin, end):
    max_fittness = 0
    best_string = None
    
    # Convert input and output to NumPy arrays if they are not already
    if not isinstance(input, np.ndarray):
        input = np.array(input)
    if not isinstance(output, np.ndarray):
        output = np.array(output)

    # Iterate through the possible hexadecimal strings with a progress bar
    for i in tqdm(range(begin, end), desc="Checking all options"):
        hex_string = transform_decimal_to_base6(i)
        binary_dict = hex_to_binary_dict(hex_string)
        fitness = check_one_option(binary_dict, input, output)
        
        if fitness > max_fittness:
            max_fittness = fitness
            best_string = hex_string

    return max_fittness, best_string


# %%

(input_train, output_train), (input_control, output_control) = create_input_output(red_wine, train_split=0.8)

max_fitness, best_string = iterate_over_possible_dict(input_train, output_train, 0, 6**16-1)
print(max_fitness, best_string)
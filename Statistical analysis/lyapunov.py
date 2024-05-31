import numpy as np
import random
from icecream import ic
import pandas as pd


def calculate_lyapnov(original, adapted):
    # Calculate the Hamming distance
    hamming_distance = sum(o != a for o, a in zip(original, adapted))
    
    # Check for zero Hamming distance to avoid log(0)
    if hamming_distance == 0:
        return 0
    
    # Calculate the natural logarithm of the Hamming distance
    log_hamming = np.log(hamming_distance)
    
    # Normalize the logarithm result by dividing by 10
    result = log_hamming / 10
    
    return result

def get_neighborhood(state, index):
    # Assuming a 1D cellular automaton with wrap-around (circular) boundaries
    left = state[index - 1] if index > 0 else state[-1]
    current = state[index]
    right = state[index + 1] if index < len(state) - 1 else state[0]
    # Convert tuple of binary digits (left, current, right) to a single integer
    neighborhood_index = left * 4 + current * 2 + right * 1  # Binary to decimal conversion
    return neighborhood_index


def evolve(seed, ruleset, steps):
    # Run the cellular automaton
    automaton_state = seed
    for step in range(steps):  # Assuming STEPS is the length of input
        new_state = [0] * CA_SIZE
        for i in range(CA_SIZE):
            rule_index = ruleset[i]
            neighbors = (automaton_state[(i-1) % CA_SIZE], automaton_state[i], automaton_state[(i+1) % CA_SIZE])
            new_state[i] = apply_rule(rule_index, neighbors)  # This function needs to be defined
        automaton_state = new_state
    return automaton_state

def apply_rule(rule, neighbors):
    # Convert the rule number to a binary string, padded to 8 bits
    rule_bin = format(rule, '08b')[::-1]  # Reverse to match index with position

    # Convert the neighbor tuple to a single integer
    neighbor_index = neighbors[0] * 4 + neighbors[1] * 2 + neighbors[2] * 1

    # Apply the rule: convert the relevant bit back to integer (0 or 1)
    next_state = int(rule_bin[neighbor_index])

    return next_state

def find_last_states(ruleset, steps=100, simulations = 10):
    results = []
    for _ in range(simulations):
        seed_length = len(ruleset)
        original_seed = [random.randint(0, 1) for _ in range(seed_length)]
        original = evolve(original_seed, ruleset, steps)
        
        # Flip the middle cell
        adapted_seed = original_seed[:]
        mid_index = len(adapted_seed) // 2
        adapted_seed[mid_index] = 1 - adapted_seed[mid_index]
        adapted = evolve(adapted_seed, ruleset, steps)
        

    return original, adapted

def langton_parameter_eca(rule_numbers):
    """
    Calculate the Langton parameter for a given list of ECA rules represented as integers from 0 to 255.
    Each integer represents a unique rule configuration for an elementary cellular automaton.

    :param rule_numbers: List of integers, each an ECA rule number from 0 to 255.
    :return: The Langton parameter as a float.
    """
    total_transitions = 0
    active_transitions = 0

    for rule_number in rule_numbers:
        # Convert the rule number to an 8-bit binary string
        rule_binary = f'{rule_number:08b}'
        # Count the number of '1's in the binary string
        active_transitions += sum(1 for bit in rule_binary if bit == '1')
        total_transitions += len(rule_binary)  # Each rule contributes 8 transitions

    if total_transitions == 0:
        ic(rule_numbers)
        return 0  # Avoid division by zero if somehow there are no transitions

    lambda_value = active_transitions / total_transitions
    return lambda_value



def create_matrices(CA_SIZE, STEPS, SIMULATIONS):
    
    #Langton
    langton_matrix = np.zeros((255,255))

    for i in range(255):
        #Define ruleset
        ruleset = [i] * CA_SIZE
        for j in range(255):
            middle = int(np.floor(CA_SIZE/2))
            ruleset[middle] = j

            #Some calculations are analytical and do not require simulations
            
            #Calculate Langton for NUCA
            langton = langton_parameter_eca(ruleset)
            langton_matrix[i,j] = langton

        print(f'we are performing calculation {i}')
    return langton_matrix

def save_data(data, name):
    df = pd.DataFrame(data)
    df.to_csv('D:\\PythonProjects\\Thesis-data\\smallCA\\Analysis\\'+ str(name) + '.csv', index=False)
    
if __name__ == '__main__':
    CA_SIZE = 51
    STEPS = 25
    SIMULATIONS = 25
    langton_matrix= create_matrices(CA_SIZE, STEPS, SIMULATIONS)
    save_data(langton_matrix, 'langton_matrix')
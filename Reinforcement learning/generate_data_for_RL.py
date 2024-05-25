import pandas as pd
import numpy as np

STEPS = 13
CA_SIZE = 25
train_split = 0.5

red_wine = pd.read_csv('red_wine.csv')


def create_input_output(red_wine, train_split=0.8):
    # Sample rows from the DataFrame
    selected_rows = red_wine.sample(n=200)  # Randomly select 200 rows

    # Calculate column averages for specified columns
    averages = selected_rows[['alcohol', 'volatile acidity', 'citric acid', 'sulphates']].mean()

    # Binary transformation based on the average
    binary_transformed = selected_rows[['alcohol', 'volatile acidity', 'citric acid', 'sulphates']].ge(averages).astype(int)

    # Keep the binary rows as lists of binary values
    input_list = binary_transformed.values.tolist()

    # Quality mapped to four-bit binary lists
    output_list = selected_rows['quality'].apply(lambda x: list(map(int, f"{x:04b}"))).tolist()

    # Shuffle the data while keeping input and output aligned
    combined_list = list(zip(input_list, output_list))
    np.random.shuffle(combined_list)
    shuffled_input, shuffled_output = zip(*combined_list)
    split_index = int(len(shuffled_input) * train_split)  # Split based on train_split percentage

    # Split data into training and control
    input_train = shuffled_input[:split_index]
    output_train = shuffled_output[:split_index]
    input_control = shuffled_input[split_index:]
    output_control = shuffled_output[split_index:]

    return (input_train, output_train), (input_control, output_control)

    return (input_train, output_train), (input_control, output_control)


def calculate_performance(state, inputs, outputs):
    fitness_scores = []

    for input, expected_output in zip(inputs, outputs):
        # Initialize the automaton's state
        automaton_state = [0] * 25  # Assuming 25 cells wide
        # Set specified cells based on the current input
        automaton_state[5] = input[0]
        automaton_state[10] = input[1]
        automaton_state[15] = input[2]
        automaton_state[20] = input[3]

        # Run the cellular automaton
        for step in range(len(input)):  # Assuming STEPS is the length of input
            new_state = [0] * 25
            for i in range(25):
                rule_index = state[i]
                neighbors = (automaton_state[(i-1) % 25], automaton_state[i], automaton_state[(i+1) % 25])
                new_state[i] = apply_rule(rule_index, neighbors)  # This function needs to be defined
            automaton_state = new_state

        # Compare automaton output to expected output
        distance = sum(1 for i in [5, 10, 15, 20] if automaton_state[i] != expected_output[[5, 10, 15, 20].index(i)])
        fitness = 1 / (1 + distance)
        fitness_scores.append(fitness)

    # Calculate average fitness
    average_fitness = sum(fitness_scores) / len(fitness_scores)
    return average_fitness

def apply_rule(rule, neighbors):
    # Convert the rule number to a binary string, padded to 8 bits
    rule_bin = format(rule, '08b')[::-1]  # Reverse to match index with position

    # Convert the neighbor tuple to a single integer
    neighbor_index = neighbors[0] * 4 + neighbors[1] * 2 + neighbors[2] * 1

    # Apply the rule: convert the relevant bit back to integer (0 or 1)
    next_state = int(rule_bin[neighbor_index])

    return next_state


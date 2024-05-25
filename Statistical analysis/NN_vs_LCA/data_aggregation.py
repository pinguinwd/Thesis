import os
import csv
import uuid
from pathlib import Path
import re

keywords = ['Best Fitness Training:', 'Best Fitness Control:','Training data set:','Control data set:']

def extract_numbers_from_file(file_path):
    """Extract numbers from the given file based on a keyword indicating the preceding line."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    output = []
    capture_next_line = False  # Flag to determine when to capture numbers
    
    for line in lines:
        if capture_next_line:
            # Extract numbers from the next line after the keyword
            numbers = [float(num) for num in re.findall(r"[-+]?\d*\.\d+|\d+", line)]
            output.append(numbers)
            capture_next_line = False
            # Assuming only one block per keyword is present; remove break if multiple blocks are expected
        if any(keyword in line for keyword in keywords):
            print(next(keyword for keyword in keywords if keyword in line))
            capture_next_line = True  # Set flag to capture numbers on the next line

    return output


def create_csv_from_data(base_folder):
    categories = ['LCA', 'NN']
    data_lists = {}  # Use to store lists of data for each column
    header = []

    # Traverse each category folder
    for category in categories:
        folder_path = Path(base_folder) / category
        counter = 1
        # Traverse each subfolder within the category
        for subfolder in folder_path.iterdir():
            if subfolder.is_dir():
                summary_file = subfolder / '_summary.txt'
                if summary_file.exists():
                    # Extract data
                    data = extract_numbers_from_file(summary_file)
                    print('data0: ', data[0])
                    print('data1: ', data[1])
                    # Create column names based on the category and a counter
                    training_col_name = f"{category}_Training_{counter}"
                    control_col_name = f"{category}_Control_{counter}"
                    header.extend([training_col_name, control_col_name])

                    # Store each set of numbers in the appropriate list
                    if training_col_name not in data_lists:
                        data_lists[training_col_name] = []
                    if control_col_name not in data_lists:
                        data_lists[control_col_name] = []
                    data_lists[training_col_name].extend(data[0])
                    data_lists[control_col_name].extend(data[1])

                    counter += 1

    # Write to CSV, creating rows for each index in the longest list
    csv_file_path = Path(base_folder) / 'compiled_data.csv'
    max_length = max(len(lst) for lst in data_lists.values())  # Find the longest list
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        for i in range(max_length):
            row = {key: data_lists[key][i] if i < len(data_lists[key]) else '' for key in header}
            writer.writerow(row)


# Example usage
create_csv_from_data('D:\\PythonProjects\\Thesis\\simulations')

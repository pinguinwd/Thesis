import os
import json
import csv
import glob
import zipfile


def calculate_growth_rate(values):
    if len(values) < 25:
        return None  # Discards data with less than 25 numbers
    first_value = values[0]
    last_value = values[-1]
    growth_rate = (last_value - first_value) / ((1 - first_value) * len(values))
    return growth_rate


def parse_summary_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.read().split('\n\n')  # Split by empty line
        ga_params = json.loads(lines[0].split(':', 1)[1].strip())
        mean_fitness_training = list(map(float, lines[1].split(':')[1].strip().split()))
        mean_fitness_control = list(map(float, lines[2].split(':')[1].strip().split()))
        best_fitness_training = list(map(float, lines[3].split(':')[1].strip().split()))
        best_fitness_control = list(map(float, lines[4].split(':')[1].strip().split()))

        return ga_params, mean_fitness_training, mean_fitness_control, best_fitness_training, best_fitness_control


def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)




def extract_and_remove_zip_files(folder_path):
    # Navigate to the folder
    os.chdir(folder_path)

    # Find all zip files
    zip_files = glob.glob('*.zip')

    for zip_file in zip_files:
        # Extract the base name to create a folder name
        folder_name = os.path.splitext(zip_file)[0]

        # Create a folder with the same name as the zip file
        os.makedirs(folder_name, exist_ok=True)

        # Extract the zip file into the folder
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(folder_name)

        # Remove the zip file after extraction
        os.remove(zip_file)
        print(f'Extracted and removed {zip_file}')


# Example usage



def main():
    base_path = 'D:\\PythonProjects\\Thesis\\GA_CA_Simulations\\Data_OtherClassifications'
    output_file = 'summary_data_other_classifications.csv'

    extract_and_remove_zip_files(base_path)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['crossover_type', 'crossover_rate', 'mutation_type', 'mutation_rate', 'mutation_std_dev', 'selection_type', 'selection_size',
                      'distance', 'case_study', 'CA_size', 'population_size', 'input_locations', 'input_order',
                      'input_percentage', 'steps', 'number_of_rules', 'sort_rules', 'chromosome_evol', 'input_loc_evol', 'output_loc_evol',
                      'growth_rate_mean_fitness_training', 'growth_rate_mean_fitness_control',
                      'growth_rate_best_fitness_training', 'growth_rate_best_fitness_control']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for root, dirs, files in os.walk(base_path):
            for file in files:
                if file.endswith('_summary.txt'):
                    file_path = os.path.join(root, file)
                    ga_params, mft, mfc, bft, bfc = parse_summary_file(file_path)

                    if any(lst[0] == 1 or len(lst) < 10 for lst in [mft, mfc, bft, bfc]):
                        continue

                    # Flatten the ga_params dictionary
                    flattened_ga_params = flatten_dict(ga_params)

                    # Prepare data row
                    row = {**flattened_ga_params,
                           'growth_rate_mean_fitness_training': calculate_growth_rate(mft),
                           'growth_rate_mean_fitness_control': calculate_growth_rate(mfc),
                           'growth_rate_best_fitness_training': calculate_growth_rate(bft),
                           'growth_rate_best_fitness_control': calculate_growth_rate(bfc)}
                    writer.writerow(row)


if __name__ == '__main__':
    main()

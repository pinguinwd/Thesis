import os
import numpy as np
import pandas as pd
import re
from ast import literal_eval
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


# Initialize the DataFrame
df = pd.DataFrame(index=range(100), columns=range(256)).fillna(0)

# Regular expression to capture the chromosome data
regex = re.compile(r"{'chromosome': array\((\[.*?\])\)", re.DOTALL)

def process_chromosome_file(file_path, df):
    with open(file_path, 'r') as file:
        content = file.read()
        matches = regex.findall(content)
        for i, match in enumerate(matches):
            if i >= 100:
                break
            chromosome_list = literal_eval(match.replace('\n', ''))
            for number in chromosome_list:
                if 0 <= number < 256:
                    df.at[i, number] += 1
    return df

def process_directory(directory, df):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('_chromosomes.txt'):
                df = process_chromosome_file(os.path.join(root, file), df)
    return df

# Initialize the DataFrame
df = pd.DataFrame(index=range(101), columns=range(256)).fillna(0)

# Process each main directory
df = process_directory('Data_MNUECA', df)
df = process_directory('Data_OtherClassifications', df)

import pandas as pd
from scipy.stats import ttest_1samp
from statsmodels.stats.multitest import multipletests


def process_dataframe(df):
    # Normalize each cell by the sum of its row
    df = df.div(df.sum(axis=1), axis=0)

    # Each cell divided by the value above (except the first row)
    df = df.divide(df.shift(1), axis=0)
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    # Define a function to perform t-tests and corrections
    def perform_tests(df, alternative='two-sided'):
        p_values = []
        columns = []
        for column in df.columns:
            t_stat, p_value = ttest_1samp(df[column][1:], 1, nan_policy='omit')  # skip the first row as it will be NaN
            if alternative == 'greater' and t_stat < 0:
                p_value = 1  # Not significant if test statistic is negative for 'greater'
            elif alternative == 'less' and t_stat > 0:
                p_value = 1  # Not significant if test statistic is positive for 'less'
            elif alternative in ['greater', 'less']:
                p_value /= 2  # Adjust p-value for one-sided tests
            p_values.append(p_value)
            columns.append(column)

        # Apply Benjamini-Hochberg correction
        reject, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
        return pd.DataFrame({
            'Column': columns,
            'P-Value': corrected_p_values,
            'Reject Null': reject,
            'Test Type': alternative
        })

    # Perform all three tests
    results_two_sided = perform_tests(df, 'two-sided')
    results_one_sided_greater = perform_tests(df, 'greater')
    results_one_sided_less = perform_tests(df, 'less')

    # Combine results into a single DataFrame
    combined_results = pd.concat([results_two_sided, results_one_sided_greater, results_one_sided_less])

    # Sort results by p-value in ascending order
    combined_results.sort_values(by=['Test Type', 'P-Value'], inplace=True)


    # Save results to a text file
    combined_results.to_csv('benjamini_test_rules.txt', sep='\t', index=False)

    return combined_results


# Example usage:
# Assuming 'df' is your DataFrame
#result_df = process_dataframe(df)
#print(result_df)



# Rule classifications by Wolfram classes and creativity
wolfram_classes = {
    'class I': [0, 8, 32, 40, 128, 136, 160, 168],
    'class II': [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 19, 23, 24, 25, 26, 27, 28, 29, 33, 34, 35, 36, 37, 38, 42, 43, 44, 46, 50, 51, 56, 57, 58, 62, 72, 73, 74, 76, 77, 78, 94, 104, 108, 130, 132, 134, 138, 140, 142, 152, 154, 156, 162, 164, 170, 172, 178, 184, 200, 204, 232],
    'class III': [18, 22, 30, 45, 60, 90, 105, 122, 126, 146, 150],
    'class IV': [41, 54, 106, 110]
}

def process_dataframe2(df):
    # Map each rule to its classification and sum occurrences
    class_sums = {key: df.loc[:, df.columns.isin(values)].sum(axis=1) for key, values in wolfram_classes.items()}

    # Create a new DataFrame from the summed occurrences
    df_classified = pd.DataFrame(class_sums)

    # Normalize each cell by the sum of its row
    df_classified = df_classified.div(df_classified.sum(axis=1), axis=0)

    # Each cell divided by the value above (except the first row)
    df_classified = df_classified.divide(df_classified.shift(1), axis=0)
    df_classified.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    
    # Statistical Analysis as previously described
    results = {}
    for column in df_classified.columns:
        t_stat, p_value = ttest_1samp(df_classified[column], 1, nan_policy='omit')
        results[column] = p_value

    # Apply Benjamini-Hochberg correction
    p_values = list(results.values())
    _, corrected_p_values, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')

    # Creating DataFrame from the t-test results with corrected p-values
    results_df = pd.DataFrame({
        'Classification': list(results.keys()),
        'P-Value': corrected_p_values
    })

    # Sort results by p-value in ascending order
    results_df.sort_values(by='P-Value', inplace=True)

    # Save results to a text file
    results_df.to_csv('benjamini_test_rules_classified.txt', sep='\t', index=False)

    return results_df

# Example usage:
# Assuming 'df' is your DataFrame with columns as rule numbers
result_df = process_dataframe2(df)
print(result_df)
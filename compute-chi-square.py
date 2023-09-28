import pandas as pd
from scipy.stats import chisquare
from tabulate import tabulate

df = pd.read_csv('./results_with_confidence_intervals.csv')
df_sorted = df.sort_values(by=['Method', 'Dataset', 'Sampling Number'])

# Compute the average accuracy(mean) for every combination of 'Method' and 'Dataset'
average_accuracy = df_sorted.groupby(['Method', 'Dataset'])['Accuracy(mean)'].mean().reset_index()

# Compute the Pearson's chi-square test for each combination

def compute_chi_square(group):
    observed = group['Accuracy(mean)']
    expected = group['Accuracy(mean)'].mean()
    chi2, _ = chisquare(observed, [expected]*len(observed))
    return round(chi2, 6)

chi_square_values = df_sorted.groupby(['Method', 'Dataset']).apply(compute_chi_square).reset_index()
chi_square_values.columns = ['Method', 'Dataset', 'Chi-Square Value']

# 5. Export the chi-square values to a CSV file
output_path = "./chi_square_values.csv"
chi_square_values.to_csv(output_path, index=False)


def print_latex_table(df, label, caption):
    """
    Print the LaTeX code for a table given a DataFrame, label, and caption.

    Parameters:
    - df (pd.DataFrame): The input data.
    - label (str): The label for the LaTeX table.
    - caption (str): The caption for the LaTeX table.
    """

    # Convert the DataFrame to a LaTeX table string
    latex_str = df.to_latex(index=False)

    # Add the label and caption
    table_code = f"""
\\begin{{table}}[h!]
\\centering
{latex_str}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
    """

    print(table_code)

print_latex_table(chi_square_values, "tab:chi_square", "Chi-square values for Method and Dataset.")

# Compute the mean chi-square value for each method
mean_chi_square = chi_square_values.groupby('Method')['Chi-Square Value'].mean().round(6).reset_index()
mean_chi_square_output_path = "./mean_chi_square_values.csv"
mean_chi_square.to_csv(mean_chi_square_output_path, index=False)
print(mean_chi_square)

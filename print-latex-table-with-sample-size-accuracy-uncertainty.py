import pandas as pd
from itertools import groupby


def sort_results(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(by=["Method", "Dataset", "Sampling Number"])


def generate_latex_tables_with_method_names(df: pd.DataFrame, max_methods_per_table=2) -> list:
    unique_methods = df['Method'].unique()
    dfs_split = [df[df['Method'].isin(unique_methods[i:i + max_methods_per_table])] for i in
                 range(0, len(unique_methods), max_methods_per_table)]
    latex_tables = []

    for df_split in dfs_split:
        rows = df_split.values.tolist()
        latex_rows = []
        last_dataset = None
        for method, group in groupby(rows, lambda x: x[0]):
            group = list(group)
            first_row = group[0]
            latex_rows.append(f"\\multirow{{{len(group)}}}{{*}}{{\\textbf{{{first_row[0]}}}}}")
            for row in group:
                if last_dataset and last_dataset != row[1]:
                    latex_rows[-1] += "\n\\cline{2-8}"
                latex_rows[-1] += " & \\textbf{" + row[1] + "}" + " & " + " & ".join(map(str, row[2:])) + " \\\\"
                last_dataset = row[1]
            latex_rows[-1] += "\n\\hline"

        methods_in_this_table = " and ".join(df_split['Method'].unique())
        latex_table_content = "\n".join(latex_rows)
        latex_table = f"""
\\begin{{table}}[h]
\\centering
\\tiny
\\begin{{tabular}}{{|l|l|r|r|r|r|r|r|}}
\\hline
\\textbf{{Method}} & \\textbf{{Dataset}} & \\textbf{{Sampling Number}} & \\textbf{{Accuracy(mean)}} & \\textbf{{Accuracy(std)}} & \\textbf{{Standard Error}} & \\textbf{{CI Lower}} & \\textbf{{CI Upper}} \\\\
\\hline
{latex_table_content}
\\end{{tabular}}
\\caption{{Comparison of methods across different datasets and sampling numbers for {methods_in_this_table}. Each experiment was repeated five times. We report the mean and standard deviation of the F1-scores of the node classification results.}}
\\label{{tab:method_comparison_part_{len(latex_tables) + 1}}}
\\end{{table}}
"""
        latex_tables.append(latex_table)

    return latex_tables


# Remap methods and datasets
remap_methods = {
    'ladies': 'LADIES',
    'fastgcn': 'FastGCN',
    'graphsaint': 'GraphSAINT',
    'gsgf': 'GRAPES',
    'asgcn': 'AS-GCN',
    'grapes': 'GRAPES',
}

remap_datasets = {
    'cora': 'Cora',
    'citeseer': 'Citeseer',
    'pubmed': 'Pubmed',
    'reddit': 'Reddit',
    'flickr': 'Flickr',
    'yelp': 'Yelp',
    'arxiv': 'ogbn-arxiv',
    'ogb-arxiv': 'ogbn-arxiv',
    'ogbn-arxiv': 'ogbn-arxiv',
    'products': 'ogbn-products',
    'ogb-products': 'ogbn-products',
    'ogbn-products': 'ogbn-products',
}

# Load the data, standardize the spelling, sort it, and round it
df = pd.read_csv("./results_with_confidence_intervals.csv")
df['Method'] = df['Method'].map(remap_methods).fillna(df['Method'])
df['Dataset'] = df['Dataset'].map(remap_datasets).fillna(df['Dataset'])
df_sorted = sort_results(df)
df_rounded_sorted = df_sorted.round(2)

# Generate the LaTeX tables and print them
latex_tables_strings = generate_latex_tables_with_method_names(df_rounded_sorted)
for table in latex_tables_strings:
    pass
    print(table)

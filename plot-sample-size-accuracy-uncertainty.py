import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def compute_confidence_intervals(df):
    df['Standard Error'] = (df['Accuracy(std)'].astype(float) / np.sqrt(
        df['Sampling Number'].replace('All', np.nan).astype(float))).round(2)

    z_value = 1.96  # Assuming normal distribution
    df['CI Lower'] = (df['Accuracy(mean)'].astype(float) - z_value * df['Standard Error']).round(2)
    df['CI Upper'] = (df['Accuracy(mean)'].astype(float) + z_value * df['Standard Error']).round(2)

    return df


dataframe = pd.read_csv("./results-uncertainty.csv")
dataframe_with_ci = compute_confidence_intervals(dataframe)

output_path = "./results_with_confidence_intervals.csv"
dataframe_with_ci.to_csv(output_path, index=False)

plt.style.use('science')

dataframe_corrected = pd.read_csv("./results_with_confidence_intervals.csv")

remap_methods = {
    'ladies': 'LADIES',
    'fastgcn': 'FastGCN',
    'graphsaint': 'GraphSAINT',
    'grapes': 'GRAPES',
    'asgcn': 'AS-GCN',
    'gfn': 'GRAPES-GFlowNet',
    'rl': 'GRAPES-RL',
    'random': 'Random',
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

colrs = {
    'LADIES': 'blue',
    'FastGCN': 'green',
    'GraphSAINT': 'orange',
    'AS-GCN': 'red',
    'GRAPES-GFlowNet': '#9EF8EE',
    'GRAPES-RL': '#45214A',
    'Random': '#747E7E',
}

data = {}
for index, row in dataframe_corrected.iterrows():
    model = remap_methods[row['Method'].lower()]
    dataset = remap_datasets[row['Dataset'].lower()]
    smaple = row['Sampling Number']
    accuracy = float(row['Accuracy(mean)'])
    ci_lower = float(row['CI Lower'])
    ci_upper = float(row['CI Upper'])

    if model not in data:
        data[model] = {}
    if dataset not in data[model]:
        data[model][dataset] = {}

    data[model][dataset][smaple] = (accuracy, ci_lower, ci_upper)

ordered_datasets = ['Reddit', 'Flickr', 'Yelp', 'ogbn-arxiv']

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for ax, dataset in zip(axes.flatten(), ordered_datasets):
    for model in ['LADIES', 'FastGCN', 'GraphSAINT', 'AS-GCN', 'Random']:
        if dataset in data.get(model, {}):
            sorted_samples = sorted(data[model][dataset].keys())
            x = []
            y = []
            lower_bound = []
            upper_bound = []
            for sample in sorted_samples:
                values = data[model][dataset][sample]
                x.append(sample)
                y.append(values[0])
                lower_bound.append(values[1])
                upper_bound.append(values[2])
            ax.plot(x, y, label=model, marker='o', color=colrs[model], linewidth=2.0, linestyle='dotted')
            ax.fill_between(x, lower_bound, upper_bound, color=colrs[model], alpha=0.04)

    for model in ['GRAPES-RL', 'GRAPES-GFlowNet']:
        if dataset in data.get(model, {}):
            sorted_samples = sorted(data[model][dataset].keys())
            x = []
            y = []
            lower_bound = []
            upper_bound = []
            for sample in sorted_samples:
                values = data[model][dataset][sample]
                x.append(sample)
                y.append(values[0])
                lower_bound.append(values[1])
                upper_bound.append(values[2])
            ax.plot(x, y, label=model, marker='o', color=colrs[model], linewidth=2.0, linestyle='solid')
            ax.fill_between(x, lower_bound, upper_bound, color=colrs[model], alpha=0.04)

    ax.set_title(f"{dataset}", fontsize=15, fontweight='bold')
    if dataset == 'Reddit':
        ax.legend(fontsize=8)
    ax.set_xticks([32, 64, 128, 256, 512])
    ax.set_xticklabels([32, 64, 128, 256, 512], rotation=90, fontsize=10)
    ax.set_xscale('log', base=2)

    # # Adjust the spines to create a gap
    # # Hide the top and right spines
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    #
    # # Move the bottom and left spines away from zero
    # ax.spines['bottom'].set_position(('outward', 10))  # Move bottom spine down
    # ax.spines['left'].set_position(('outward', 10))  # Move left spine left

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.1)
fig.text(0.01, 0.5, 'Node Classification F1-scores', va='center', rotation='vertical', fontsize=15, fontweight='bold')
fig.text(0.5, -0.05, 'Sample Size (logarithmic base-2)', ha='center', fontsize=15, fontweight='bold')

plt.savefig("sampling_vs_accuracy_confidence_intervals.pdf", format='pdf', bbox_inches='tight')
plt.show()

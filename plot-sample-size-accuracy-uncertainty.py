import pandas as pd
import matplotlib.pyplot as plt

# Set the style
plt.style.use('science')

# Read the corrected CSV file
dataframe_corrected = pd.read_csv("./results_with_confidence_intervals.csv")

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

# Colors for models
colors = {
    'LADIES': 'black',
    'FastGCN': 'green',
    'GraphSAINT': 'orange',
    'GRAPES': '#4040FF',
    'AS-GCN': 'red',
}

# Create a structured data dictionary from the corrected dataframe
data = {}
for index, row in dataframe_corrected.iterrows():
    model = remap_methods[row['Method'].lower()]
    dataset = remap_datasets[row['Dataset'].lower()]
    sample = row['Sampling Number']
    accuracy = float(row['Accuracy(mean)'])
    ci_lower = float(row['CI Lower'])
    ci_upper = float(row['CI Upper'])

    if model not in data:
        data[model] = {}
    if dataset not in data[model]:
        data[model][dataset] = {}

    data[model][dataset][sample] = (accuracy, ci_lower, ci_upper)

# Ordered datasets
ordered_datasets = ['Cora', 'Citeseer', 'Pubmed', 'Reddit', 'Flickr', 'Yelp', 'ogbn-arxiv', 'ogbn-products']

# Plotting
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
for ax, dataset in zip(axes.flatten(), ordered_datasets):
    for model, datasets in data.items():
        if dataset in datasets:
            x = []
            y = []
            lower_bound = []
            upper_bound = []
            for sample, values in datasets[dataset].items():
                x.append(sample)
                y.append(values[0])
                lower_bound.append(values[1])
                upper_bound.append(values[2])
            ax.plot(x, y, label=model, marker='o', color=colors[model], linewidth=2.0)
            ax.fill_between(x, lower_bound, upper_bound, color=colors[model], alpha=0.2)

    ax.set_title(f"{dataset}", fontsize=20, fontweight='bold')
    if dataset == 'Reddit':
        ax.legend()
    ax.grid(True)
    ax.set_xticks([32, 64, 128, 256, 512])
    ax.set_xticklabels([32, 64, 128, 256, 512], rotation=90, fontsize=10)
    ax.set_ylim(0, 100)

    # Set the x-axis to be logarithmic with base 2
    ax.set_xscale('log', base=2)

plt.tight_layout()

# Padding around the figure
plt.subplots_adjust(left=0.05, right=0.99, top=0.94, bottom=0.1)

# Add a common y-axis label and x-axis label to the entire grid of subplots
fig.text(0.01, 0.5, 'Node Classification F1-scores', va='center', rotation='vertical', fontsize=15, fontweight='bold')
fig.text(0.5, 0.02, 'Sample Size (logarithmic base-2)', ha='center', fontsize=15, fontweight='bold')

plt.savefig("sampling_vs_accuracy_confidence_intervals.pdf", format='pdf', bbox_inches='tight')
plt.show()

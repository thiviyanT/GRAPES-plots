import pandas as pd
import matplotlib.pyplot as plt

# Set the style
plt.style.use('science')

# Read the corrected CSV file
dataframe_corrected = pd.read_csv("./results_with_confidence_intervals.csv")

# Remap methods and datasets
remap_methods = {
    'ladies': 'Ladies',
    'fastgcn': 'FastGCN',
    'graphsaint': 'GraphSAINT',
    'gsgf': 'GRAPES',
    'asgcn': 'AS-GCN',
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
    'Ladies': 'black',
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

plt.tight_layout()
plt.savefig("sampling_vs_accuracy.pdf", format='pdf', bbox_inches='tight')
plt.show()

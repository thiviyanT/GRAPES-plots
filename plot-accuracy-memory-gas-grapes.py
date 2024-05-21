import matplotlib.pyplot as plt
import numpy as np

# Use the 'science' style for cleaner plots
plt.style.use('science')

# Data extracted from the table
datasets = ["Cora", "Citeseer", "Pubmed", "Reddit", "Flickr", "Yelp", "ogbn-arxiv", "ogbn-products"]
methods = ["GRAPES-32", "GRAPES-256", "GAS"]

# Accuracy and Memory values for each method and dataset
accuracy = {
    "GRAPES-32": [88.10, 79.04, 89.58, 92.43, 45.69, 44.50, 64.03, 73.62],
    "GRAPES-256": [87.29, 78.74, 90.11, 93.68, 47.33, 44.91, 64.54, 73.65],
    "GAS": [87.67, 79.37, 87.94, 94.83, 51.32, 33.79, 69.38, 75.12]
}
memory = {
    "GRAPES-32": [42.02, 95.15, 22.72, 321.78, 47.01, 81.78, 44.96, 362.54],
    "GRAPES-256": [49.06, 106.68, 42.07, 590.90, 134.53, 242.42, 127.89, 540.90],
    "GAS": [46.43, 117.85, 184.46, 4052.64, 845.06, 7144.45, 1191.78, 21609.92]
}

colors = {
    "GRAPES-32": "#8E1ED9",  # Purple shade for GRAPES-32
    "GRAPES-256": "#590D8C",  # Darker purple shade for GRAPES-256
    "GAS": "#27ae60"  # Green shade for GAS
}

legend_font_size = 20
axis_font_size = 18
tick_font_size = 18
xtick_rotation = 30

# Plotting
barWidth = 0.25
r1 = np.arange(len(accuracy["GRAPES-32"]))  # the label locations
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

fig, ax = plt.subplots(1, 2, figsize=(18, 4.5))

# F1-Scores (Accuracy) subplot
ax[0].bar(r1, accuracy["GRAPES-32"], width=barWidth, edgecolor='white', color=colors["GRAPES-32"], label='GRAPES-32')
ax[0].bar(r2, accuracy["GRAPES-256"], width=barWidth, edgecolor='white', color=colors["GRAPES-256"], label='GRAPES-256')
ax[0].bar(r3, accuracy["GAS"], width=barWidth, edgecolor='white', color=colors["GAS"], label='GAS')
# ax[0].set_title('F1-Scores (Accuracy) Comparison', fontweight='bold')
# ax[0].set_xlabel('Datasets', fontweight='bold', fontsize=axis_font_size)
ax[0].set_ylabel('F1-Score', fontweight='bold', fontsize=axis_font_size)
ax[0].set_ylim(0, 100)
ax[0].set_xticks([r + barWidth for r in range(len(accuracy["GRAPES-32"]))])
ax[0].set_xticklabels(datasets, rotation=xtick_rotation)
ax[0].tick_params(axis='x', labelsize=tick_font_size)
ax[0].tick_params(axis='y', labelsize=tick_font_size)
# ax[0].legend(fontsize=legend_font_size)

# Memory Allocation subplot with logarithmic y-axis
ax[1].bar(r1, memory["GRAPES-32"], width=barWidth, edgecolor='white', color=colors["GRAPES-32"], label='GRAPES-32', log=True)
ax[1].bar(r2, memory["GRAPES-256"], width=barWidth, edgecolor='white', color=colors["GRAPES-256"], label='GRAPES-256', log=True)
ax[1].bar(r3, memory["GAS"], width=barWidth, edgecolor='white', color=colors["GAS"], label='GAS', log=True)
# ax[1].set_title('Memory Allocation Comparison (Logarithmic Scale)', fontweight='bold')
# ax[1].set_xlabel('Datasets', fontweight='bold', fontsize=axis_font_size)
ax[1].set_ylabel('Memory (MB) - Log Scale', fontweight='bold', fontsize=axis_font_size)
ax[1].set_xticks([r + barWidth for r in range(len(memory["GRAPES-32"]))])
ax[1].set_xticklabels(datasets, rotation=xtick_rotation)
ax[1].tick_params(axis='x', labelsize=tick_font_size)
ax[1].tick_params(axis='y', labelsize=tick_font_size)
ax[1].legend(fontsize=legend_font_size)

plt.subplots_adjust(wspace=1.0)
plt.tight_layout()
plt.show()
fig.savefig("accuracy-memory-gas-grapes-plot.pdf", bbox_inches='tight')


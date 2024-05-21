import matplotlib.pyplot as plt
import numpy as np

# Use the 'science' style for cleaner plots
plt.style.use('science')

# Given data and configurations
datasets = ["Cora", "Citeseer", "Pubmed", "Reddit", "Flickr", "Yelp", "ogbn-arxiv", "ogbn-products"]

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

# Plotting configurations
barWidth = 0.25
r1 = np.arange(len(memory["GRAPES-32"]))  # the label locations
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

# Plotting only the Memory Allocation subplot
fig, ax = plt.subplots(figsize=(7, 4.5))

# Memory Allocation subplot with logarithmic y-axis
ax.bar(r1, memory["GRAPES-32"], width=barWidth, edgecolor='white', color=colors["GRAPES-32"], label='GRAPES-32', log=True)
ax.bar(r2, memory["GRAPES-256"], width=barWidth, edgecolor='white', color=colors["GRAPES-256"], label='GRAPES-256', log=True)
ax.bar(r3, memory["GAS"], width=barWidth, edgecolor='white', color=colors["GAS"], label='GAS', log=True)

ax.set_ylabel('Memory (MB) - Log Scale', fontweight='bold', fontsize=axis_font_size)
ax.set_xticks([r + barWidth for r in range(len(memory["GRAPES-32"]))])
ax.set_xticklabels(datasets, rotation=xtick_rotation)
ax.tick_params(axis='x', labelsize=tick_font_size)
ax.tick_params(axis='y', labelsize=tick_font_size)
ax.legend(fontsize=legend_font_size)

plt.tight_layout()
plt.show()

# Saving the plot as a PDF
memory_plot_path = "memory-gas-grapes-plot.pdf"
fig.savefig(memory_plot_path, bbox_inches='tight')

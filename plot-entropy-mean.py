import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the CSV files into DataFrames
df = pd.read_csv('csv/mean_entropy_0.csv')
df_new = pd.read_csv('csv/mean_entropy_1.csv')

# Determine the total number of steps based on the maximum step value from both datasets
overall_max_step = max(df['Step'].max(), df_new['Step'].max())

# Calculate steps per epoch to ensure 150 epochs in total
steps_per_epoch_corrected = overall_max_step / 150

# Convert the 'Step' column to 'Epoch' based on the corrected steps per epoch
df['Epoch'] = np.ceil(df['Step'] / steps_per_epoch_corrected).astype(int)
df_new['Epoch'] = np.ceil(df_new['Step'] / steps_per_epoch_corrected).astype(int)

# Compute the average mean entropy for each epoch in both datasets
avg_entropy_df = df.groupby('Epoch')['swept-blaze-817 - mean_entropy_0'].mean().reset_index()
avg_entropy_df_new = df_new.groupby('Epoch')['swept-blaze-817 - mean_entropy_1'].mean().reset_index()

# Merge the average mean entropies from both datasets based on the 'Epoch' column
avg_entropy_combined = pd.merge(avg_entropy_df, avg_entropy_df_new, on='Epoch', how='outer')

plt.style.use('science')

plt.figure(figsize=(12, 6))

plt.plot(avg_entropy_combined['Epoch'], avg_entropy_combined['swept-blaze-817 - mean_entropy_0'], 
         label='1-hop node sampling', color='blue')
plt.plot(avg_entropy_combined['Epoch'], avg_entropy_combined['swept-blaze-817 - mean_entropy_1'], 
         label='2-hop node sampling', color='green')

plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Average Mean Entropy', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0.7, 1.00)
plt.xlim(0, 150)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()

output_path = "average_entropy_vs_epoch.pdf"
plt.savefig(output_path, format='pdf')

plt.show()


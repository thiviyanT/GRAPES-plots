# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Setting the style for the plot
plt.style.use('science')

# Load the CSV files containing standard deviations into DataFrames
df_std = pd.read_csv('csv/std_entropy_0.csv')
df_std_new = pd.read_csv('csv/std_entropy_1.csv')

# Determine the total number of steps based on the maximum step value from both datasets
overall_max_step_std = max(df_std['Step'].max(), df_std_new['Step'].max())

# Calculate steps per epoch to ensure 150 epochs in total for the new datasets
steps_per_epoch_corrected_std = overall_max_step_std / 150

# Convert the 'Step' column to 'Epoch' based on the corrected steps per epoch for the new datasets
df_std['Epoch'] = np.ceil(df_std['Step'] / steps_per_epoch_corrected_std).astype(int)
df_std_new['Epoch'] = np.ceil(df_std_new['Step'] / steps_per_epoch_corrected_std).astype(int)

# Compute the average standard deviation for each epoch in both datasets
avg_std_df = df_std.groupby('Epoch')['swept-blaze-817 - std_entropy_0'].mean().reset_index()
avg_std_df_new = df_std_new.groupby('Epoch')['swept-blaze-817 - std_entropy_1'].mean().reset_index()

# Merge the average standard deviations from both datasets based on the 'Epoch' column
avg_std_combined = pd.merge(avg_std_df, avg_std_df_new, on='Epoch', how='outer')

# Plotting
plt.figure(figsize=(12, 6))

plt.plot(avg_std_combined['Epoch'], avg_std_combined['swept-blaze-817 - std_entropy_0'], 
         label='1-hop node sampling', color='blue')
plt.plot(avg_std_combined['Epoch'], avg_std_combined['swept-blaze-817 - std_entropy_1'], 
         label='2-hop node sampling', color='green')

plt.xlabel('Epoch', fontsize=20)
plt.ylabel('Standard Deviation', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylim(0.00, 0.10)  # Adjusted y-axis range
plt.xlim(0, 150)
plt.legend(fontsize=20)
plt.grid(True)
plt.tight_layout()

plt.savefig("average_std_dev_vs_epoch_adjusted.pdf", format='pdf')

plt.show()


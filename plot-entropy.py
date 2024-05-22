import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

epochs = {
    'citeseer': 300,
    'cora': 50,
    'flickr': 300,
    'ogbn-arxiv': 150,
    'obgn-products': 100,
    'pubmed': 300,
    'reddit': 50,
    'yelp': 150
}

def process_and_plot(csv_mean_0, csv_mean_1, csv_std_0, csv_std_1, output_path):
    # Setting the style
    plt.style.use('science')

    # Load mean entropy CSV files
    df = pd.read_csv(csv_mean_0)
    df_new = pd.read_csv(csv_mean_1)

    # Extract column names for mean entropy
    mean_entropy_0_col = df.columns[1]
    mean_entropy_1_col = df_new.columns[1]

    # Determine the total number of steps based on the maximum step value from both datasets
    overall_max_step = max(df['Step'].max(), df_new['Step'].max())

    # Calculate steps per epoch to ensure 150 epochs in total
    steps_per_epoch_corrected = overall_max_step / 150

    # Convert the 'Step' column to 'Epoch' for mean entropy data
    df['Epoch'] = np.ceil(df['Step'] / steps_per_epoch_corrected).astype(int)
    df_new['Epoch'] = np.ceil(df_new['Step'] / steps_per_epoch_corrected).astype(int)

    # Compute the average mean entropy for each epoch in both datasets
    avg_entropy_df = df.groupby('Epoch')[mean_entropy_0_col].mean().reset_index()
    avg_entropy_df_new = df_new.groupby('Epoch')[mean_entropy_1_col].mean().reset_index()

    # Merge the average mean entropies from both datasets
    avg_entropy_combined = pd.merge(avg_entropy_df, avg_entropy_df_new, on='Epoch', how='outer')

    # Load standard deviation CSV files
    df_std = pd.read_csv(csv_std_0)
    df_std_new = pd.read_csv(csv_std_1)

    # Extract column names for standard deviation
    std_entropy_0_col = df_std.columns[1]
    std_entropy_1_col = df_std_new.columns[1]

    # Convert the 'Step' column to 'Epoch' for standard deviation data
    df_std['Epoch'] = np.ceil(df_std['Step'] / steps_per_epoch_corrected).astype(int)
    df_std_new['Epoch'] = np.ceil(df_std_new['Step'] / steps_per_epoch_corrected).astype(int)

    # Compute the average standard deviation for each epoch in both datasets
    avg_std_df = df_std.groupby('Epoch')[std_entropy_0_col].mean().reset_index()
    avg_std_df_new = df_std_new.groupby('Epoch')[std_entropy_1_col].mean().reset_index()

    # Merge the average standard deviations from both datasets
    avg_std_combined = pd.merge(avg_std_df, avg_std_df_new, on='Epoch', how='outer')

    # Setting up subplots
    fig, ax = plt.subplots(1, 2, figsize=(24, 8))

    # Plotting mean entropy on the left subplot
    ax[0].plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_0_col],
               label='1-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='--')
    ax[0].plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_1_col],
               label='2-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='solid')
    ax[0].set_xlabel('Epoch', fontsize=40)
    ax[0].set_ylabel('Mean Entropy', fontsize=40)
    ax[0].set_xticks(ticks=range(0, 151, 20))
    ax[0].tick_params(axis='both', which='major', labelsize=20)
    ax[0].set_ylim(0.7, 1.00)
    ax[0].set_xlim(0, 150)
    ax[0].legend(fontsize=30)
    ax[0].grid(True)

    # Plotting standard deviation on the right subplot
    ax[1].plot(avg_std_combined['Epoch'], avg_std_combined[std_entropy_0_col],
               label='1-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='--')
    ax[1].plot(avg_std_combined['Epoch'], avg_std_combined[std_entropy_1_col],
               label='2-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='solid')
    ax[1].set_xlabel('Epoch', fontsize=40)
    ax[1].set_ylabel('Entropy Standard Deviation', fontsize=40)
    ax[1].set_xticks(ticks=range(0, 151, 20))
    ax[1].tick_params(axis='both', which='major', labelsize=20)
    ax[1].set_ylim(0.00, 0.10)
    ax[1].set_xlim(0, 150)
    ax[1].legend(fontsize=30)
    ax[1].grid(True)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    plt.savefig(output_path, format='pdf')
    plt.show()

def process_and_plot_together(csv_mean_0, csv_mean_1, csv_std_0, csv_std_1, output_path, ax=None):
    # Setting the style
    plt.style.use('science')

    # Load mean entropy CSV files
    df = pd.read_csv(csv_mean_0)
    df_new = pd.read_csv(csv_mean_1)

    # Extract column names for mean entropy
    mean_entropy_0_col = df.columns[1]
    mean_entropy_1_col = df_new.columns[1]

    # Determine the total number of steps based on the maximum step value from both datasets
    overall_max_step = max(df['Step'].max(), df_new['Step'].max())

    # Calculate steps per epoch to ensure 150 epochs in total
    steps_per_epoch_corrected = overall_max_step / 150

    # Convert the 'Step' column to 'Epoch' for mean entropy data
    df['Epoch'] = np.ceil(df['Step'] / steps_per_epoch_corrected).astype(int)
    df_new['Epoch'] = np.ceil(df_new['Step'] / steps_per_epoch_corrected).astype(int)

    # Compute the average mean entropy for each epoch in both datasets
    avg_entropy_df = df.groupby('Epoch')[mean_entropy_0_col].mean().reset_index()
    avg_entropy_df_new = df_new.groupby('Epoch')[mean_entropy_1_col].mean().reset_index()

    # Merge the average mean entropies from both datasets
    avg_entropy_combined = pd.merge(avg_entropy_df, avg_entropy_df_new, on='Epoch', how='outer')

    # Load standard deviation CSV files
    df_std = pd.read_csv(csv_std_0)
    df_std_new = pd.read_csv(csv_std_1)

    # Extract column names for standard deviation
    std_entropy_0_col = df_std.columns[1]
    std_entropy_1_col = df_std_new.columns[1]

    # Convert the 'Step' column to 'Epoch' for standard deviation data
    df_std['Epoch'] = np.ceil(df_std['Step'] / steps_per_epoch_corrected).astype(int)
    df_std_new['Epoch'] = np.ceil(df_std_new['Step'] / steps_per_epoch_corrected).astype(int)

    # Compute the average standard deviation for each epoch in both datasets
    avg_std_df = df_std.groupby('Epoch')[std_entropy_0_col].mean().reset_index()
    avg_std_df_new = df_std_new.groupby('Epoch')[std_entropy_1_col].mean().reset_index()

    # Merge the average standard deviations from both datasets
    avg_std_combined = pd.merge(avg_std_df, avg_std_df_new, on='Epoch', how='outer')

    if ax is None:
        ax = plt.gca()

    # Plotting mean entropy on the left subplot
    ax.plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_0_col],
               label='1-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='--')
    ax.plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_1_col],
               label='2-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='solid')
    ax.set_xlabel('Epoch', fontsize=40)
    ax.set_ylabel('Mean Entropy', fontsize=40)
    ax.set_xticks(ticks=range(0, 151, 20))
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylim(0.7, 1.00)
    ax.set_xlim(0, 150)
    ax.legend(fontsize=30)
    ax.grid(False)

    # Plotting standard deviation on the right subplot
    ax.fill_between(avg_std_combined['Epoch'],
                    avg_entropy_combined[mean_entropy_0_col] - avg_std_combined[std_entropy_0_col],
                    avg_entropy_combined[mean_entropy_0_col] + avg_std_combined[std_entropy_0_col],
               color='#4040FF')

    ax.fill_betwen(avg_std_combined['Epoch'],
               avg_entropy_combined[mean_entropy_1_col] - avg_std_combined[std_entropy_1_col],
               avg_entropy_combined[mean_entropy_1_col] - avg_std_combined[std_entropy_1_col],
               color='#4040FF')

if __name__ == '__main__':
    process_and_plot_together('csv/mean_entropy_0.csv', 'csv/mean_entropy_1.csv', 'csv/std_entropy_0.csv', 'csv/std_entropy_1.csv')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.2)

    plt.savefig('average_entropy_vs_epoch.pdf', format='pdf')
    # plt.show()

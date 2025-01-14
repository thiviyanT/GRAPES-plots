import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import scienceplots
plt.style.use('science')

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
    'blogcat': 'BlogCat',
    'snap-patents': 'snap-patents',
    'dblp': 'DBLP',
    'ogbn-proteins': 'ogbn-proteins',
}

STD_ALPHA = 0.2

def plot_single_dataset(dataset_name, datasets_info, entropy_plots_dir):
    dataset_dir = os.path.join(entropy_plots_dir, dataset_name)
    csv_mean_0_path = os.path.join(dataset_dir, 'mean_entropy0.csv')
    csv_mean_1_path = os.path.join(dataset_dir, 'mean_entropy1.csv')
    csv_std_0_path = os.path.join(dataset_dir, 'std_entropy0.csv')
    csv_std_1_path = os.path.join(dataset_dir, 'std_entropy1.csv')

    if os.path.exists(csv_mean_0_path) and os.path.exists(csv_mean_1_path) and os.path.exists(csv_std_0_path) and os.path.exists(csv_std_1_path):
        overall_max_step = max(pd.read_csv(csv_mean_0_path)['Step'].max(), pd.read_csv(csv_mean_1_path)['Step'].max())
        steps_per_epoch_corrected = overall_max_step / datasets_info[dataset_name]['epochs']
        fig, ax = plt.subplots(1, 2, figsize=(24, 7))
        process_and_plot(csv_mean_0_path, csv_mean_1_path, csv_std_0_path, csv_std_1_path, steps_per_epoch_corrected, datasets_info[dataset_name]['xlim'], ax, dataset_name)
        output_path = os.path.join(entropy_plots_dir, f"{dataset_name}-entropy-plot-single.pdf")
        plt.tight_layout()
        plt.savefig(output_path, format='pdf')
        plt.show()

def process_and_plot(csv_mean_0, csv_mean_1, csv_std_0, csv_std_1, steps_per_epoch_corrected, xlim_value, ax, dataset_name):
    df = pd.read_csv(csv_mean_0)
    df_new = pd.read_csv(csv_mean_1)
    mean_entropy_0_col = df.columns[1]
    mean_entropy_1_col = df_new.columns[1]
    df['Epoch'] = np.ceil(df['Step'] / steps_per_epoch_corrected).astype(int)
    df_new['Epoch'] = np.ceil(df_new['Step'] / steps_per_epoch_corrected).astype(int)
    avg_entropy_df = df.groupby('Epoch')[mean_entropy_0_col].mean().reset_index()
    avg_entropy_df_new = df_new.groupby('Epoch')[mean_entropy_1_col].mean().reset_index()
    avg_entropy_combined = pd.merge(avg_entropy_df, avg_entropy_df_new, on='Epoch', how='outer')
    df_std = pd.read_csv(csv_std_0)
    df_std_new = pd.read_csv(csv_std_1)
    std_entropy_0_col = df_std.columns[1]
    std_entropy_1_col = df_std_new.columns[1]
    df_std['Epoch'] = np.ceil(df_std['Step'] / steps_per_epoch_corrected).astype(int)
    df_std_new['Epoch'] = np.ceil(df_std_new['Step'] / steps_per_epoch_corrected).astype(int)
    avg_std_df = df_std.groupby('Epoch')[std_entropy_0_col].mean().reset_index()
    avg_std_df_new = df_std_new.groupby('Epoch')[std_entropy_1_col].mean().reset_index()
    avg_std_combined = pd.merge(avg_std_df, avg_std_df_new, on='Epoch', how='outer')
    mean_entropy_ymin = min(avg_entropy_combined[mean_entropy_0_col].min(), avg_entropy_combined[mean_entropy_1_col].min()) - 0.05
    mean_entropy_ymax = max(avg_entropy_combined[mean_entropy_0_col].max(), avg_entropy_combined[mean_entropy_1_col].max()) + 0.05
    std_entropy_ymin = min(avg_std_combined[std_entropy_0_col].min(), avg_std_combined[std_entropy_1_col].min()) - 0.005
    std_entropy_ymax = max(avg_std_combined[std_entropy_0_col].max(), avg_std_combined[std_entropy_1_col].max()) + 0.005

    try:
        ax[0].plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_0_col], label='1-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='--')
    except:
        print(avg_entropy_combined)

    ax[0].plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_1_col], label='2-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='solid')
    ax[0].set_xlabel('Epoch', fontsize=35)
    ax[0].set_ylabel('Mean Entropy', fontsize=35)
    ax[0].set_xticks(ticks=range(0, xlim_value + 1, xlim_value // 10))
    ax[0].tick_params(axis='both', which='major', labelsize=25)
    ax[0].set_ylim(mean_entropy_ymin, mean_entropy_ymax)
    ax[0].set_xlim(0, xlim_value)
    ax[0].legend(fontsize=40)
    ax[0].set_title(remap_datasets[dataset_name.lower()], fontsize=30)

    ax[1].plot(avg_std_combined['Epoch'], avg_std_combined[std_entropy_0_col], label='1-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='--')
    ax[1].plot(avg_std_combined['Epoch'], avg_std_combined[std_entropy_1_col], label='2-hop node sampling', color='#4040FF', linewidth=4.0, linestyle='solid')
    ax[1].set_xlabel('Epoch', fontsize=35)
    ax[1].set_ylabel('Entropy Standard Deviation', fontsize=35)
    ax[1].set_xticks(ticks=range(0, xlim_value + 1, xlim_value // 10))
    ax[1].tick_params(axis='both', which='major', labelsize=25)
    ax[1].set_ylim(std_entropy_ymin, std_entropy_ymax)
    ax[1].set_xlim(0, xlim_value)
    ax[1].legend(fontsize=40)
    ax[1].set_title(remap_datasets[dataset_name.lower()], fontsize=30)


def process_and_plot_combined(csv_mean_0, csv_mean_1, csv_std_0, csv_std_1, steps_per_epoch_corrected, xlim_value, ax,
                              dataset_name, first, left, bottom):

    """
    Plot the mean entropy with the standard deviation as a confidence region.
    """
    df = pd.read_csv(csv_mean_0)
    df_new = pd.read_csv(csv_mean_1)
    mean_entropy_0_col = df.columns[1]
    mean_entropy_1_col = df_new.columns[1]

    df['Epoch'] = np.ceil(df['Step'] / steps_per_epoch_corrected).astype(int)
    df_new['Epoch'] = np.ceil(df_new['Step'] / steps_per_epoch_corrected).astype(int)
    avg_entropy_df = df.groupby('Epoch')[mean_entropy_0_col].mean().reset_index()
    avg_entropy_df_new = df_new.groupby('Epoch')[mean_entropy_1_col].mean().reset_index()
    avg_entropy_combined = pd.merge(avg_entropy_df, avg_entropy_df_new, on='Epoch', how='outer')

    df_std = pd.read_csv(csv_std_0)
    df_std_new = pd.read_csv(csv_std_1)
    std_entropy_0_col = df_std.columns[1]
    std_entropy_1_col = df_std_new.columns[1]
    df_std['Epoch'] = np.ceil(df_std['Step'] / steps_per_epoch_corrected).astype(int)
    df_std_new['Epoch'] = np.ceil(df_std_new['Step'] / steps_per_epoch_corrected).astype(int)
    avg_std_df = df_std.groupby('Epoch')[std_entropy_0_col].mean().reset_index()
    avg_std_df_new = df_std_new.groupby('Epoch')[std_entropy_1_col].mean().reset_index()
    avg_std_combined = pd.merge(avg_std_df, avg_std_df_new, on='Epoch', how='outer')
    mean_entropy_ymin = min(avg_entropy_combined[mean_entropy_0_col].min(),
                            avg_entropy_combined[mean_entropy_1_col].min()) - 0.05
    mean_entropy_ymax = max(avg_entropy_combined[mean_entropy_0_col].max(),
                            avg_entropy_combined[mean_entropy_1_col].max()) + 0.05

    std_entropy_ymin = min(avg_std_combined[std_entropy_0_col].min(), avg_std_combined[std_entropy_1_col].min()) - 0.005
    std_entropy_ymax = max(avg_std_combined[std_entropy_0_col].max(), avg_std_combined[std_entropy_1_col].max()) + 0.005

    try:
        ax.plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_0_col], label='1-hop node sampling',
                   color='b', linewidth=4.0, linestyle='--')
    except:
        print(avg_entropy_combined)

    ax.plot(avg_entropy_combined['Epoch'], avg_entropy_combined[mean_entropy_1_col], label='2-hop node sampling',
               color='g', linewidth=4.0, linestyle='solid')

    if bottom:
        ax.set_xlabel(f'Epoch', fontsize=35)
    if left:
        ax.set_ylabel('Mean Entropy', fontsize=35)

    ax.set_xticks(ticks=range(0, xlim_value + 1, xlim_value // 10))
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, xlim_value)

    ax.set_title(remap_datasets[dataset_name.lower()], fontsize=30)

    ax.fill_between(avg_std_combined['Epoch'],
                    avg_entropy_combined[mean_entropy_0_col] - avg_std_combined[std_entropy_0_col],
                    avg_entropy_combined[mean_entropy_0_col] + avg_std_combined[std_entropy_0_col],
               color='b', alpha=STD_ALPHA, linewidth=0)
    ax.fill_between(avg_std_combined['Epoch'],
                    avg_entropy_combined[mean_entropy_1_col] - avg_std_combined[std_entropy_1_col],
                    avg_entropy_combined[mean_entropy_1_col] + avg_std_combined[std_entropy_1_col],
               color='g', alpha=STD_ALPHA, linewidth=0)

    ax.spines[['right', 'top']].set_visible(False)
    ax.tick_params(
        axis='x',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        bottom=True,  # ticks along the bottom edge are off
        top=False,  # ticks along the top edge are off
        labelbottom=True)  # labels along the bottom edge are off
    ax.tick_params(
        axis='y',  # changes apply to the x-axis
        which='both',  # both major and minor ticks are affected
        left=True,  # ticks along the bottom edge are off
        right=False,  # ticks along the top edge are off
        labelright=False)  # labels along the bottom edge are off

    if first:
        ax.legend(fontsize=30, loc='upper right' if dataset_name.lower() == 'blogcat' else 'lower left')

def plot_on_subplot(csv_mean_0, csv_mean_1, csv_std_0, csv_std_1, steps_per_epoch_corrected, xlim_value, ax, dataset_name):
    process_and_plot(csv_mean_0, csv_mean_1, csv_std_0, csv_std_1, steps_per_epoch_corrected, xlim_value, ax, dataset_name)

def generate_combined_plot(datasets_info, entropy_plots_dir):
    ordered_datasets = ['Citeseer', 'Cora', 'Flickr', 'ogbn-arxiv', 'ogbn-products', 'Pubmed', 'Reddit', 'Yelp', 'BlogCat', 'snap-patents', 'DBLP', 'ogbn-proteins']
    num_datasets = len(ordered_datasets)
    num_figures = (num_datasets + 3) // 4

    for fig_num in range(num_figures):
        start_idx = fig_num * 4
        end_idx = min((fig_num + 1) * 4, num_datasets)
        fig, axs = plt.subplots(min(4, end_idx - start_idx), 2, figsize=(24, 6 * min(4, end_idx - start_idx)))

        for idx, dataset_name in enumerate(ordered_datasets[start_idx:end_idx]):
            dataset_info = datasets_info[dataset_name]
            dataset_dir = os.path.join(entropy_plots_dir, dataset_name)
            csv_mean_0_path = os.path.join(dataset_dir, 'mean_entropy0.csv')
            csv_mean_1_path = os.path.join(dataset_dir, 'mean_entropy1.csv')
            csv_std_0_path = os.path.join(dataset_dir, 'std_entropy0.csv')
            csv_std_1_path = os.path.join(dataset_dir, 'std_entropy1.csv')

            if os.path.exists(csv_mean_0_path) and os.path.exists(csv_mean_1_path) and os.path.exists(csv_std_0_path) and os.path.exists(csv_std_1_path):
                overall_max_step = max(pd.read_csv(csv_mean_0_path)['Step'].max(), pd.read_csv(csv_mean_1_path)['Step'].max())
                steps_per_epoch_corrected = overall_max_step / dataset_info['epochs']
                plot_on_subplot(csv_mean_0_path, csv_mean_1_path, csv_std_0_path, csv_std_1_path, steps_per_epoch_corrected, dataset_info['xlim'], axs[idx], dataset_name)

        output_path = os.path.join(entropy_plots_dir, f"combined-entropy-plot-{fig_num + 1}.pdf")
        plt.tight_layout()
        plt.savefig(output_path, format='pdf')
        plt.show()

def generate_combined_plot_single(datasets_info, entropy_plots_dir, all=True):
    ordered_datasets = ['Citeseer', 'Cora', 'Flickr', 'ogbn-arxiv', 'ogbn-products', 'Pubmed', 'Reddit', 'Yelp', 'BlogCat', 'snap-patents', 'DBLP', 'ogbn-proteins'] if all else ['ogbn-products', 'DBLP']
    num_datasets = len(ordered_datasets)
    num_figures = (num_datasets) // 8

    for fig_num, start_idx in enumerate(range(0, num_datasets, 8)):

        end_idx = min(start_idx + 8, num_datasets)
        n = end_idx - start_idx

        print(n)

        rows = n // 2

        fig, axs = plt.subplots(rows, 2, figsize=(24, 6 * rows), sharex=False, sharey=True)

        if n > 2:
            axs = [ax for axrow in axs for ax in axrow] # flatten

        first = True
        for idx, dataset_name in enumerate(ordered_datasets[start_idx:end_idx]):
            dataset_info = datasets_info[dataset_name]
            dataset_dir = os.path.join(entropy_plots_dir, dataset_name)
            csv_mean_0_path = os.path.join(dataset_dir, 'mean_entropy0.csv')
            csv_mean_1_path = os.path.join(dataset_dir, 'mean_entropy1.csv')
            csv_std_0_path = os.path.join(dataset_dir, 'std_entropy0.csv')
            csv_std_1_path = os.path.join(dataset_dir, 'std_entropy1.csv')

            if os.path.exists(csv_mean_0_path) and os.path.exists(csv_mean_1_path) and os.path.exists(csv_std_0_path) and os.path.exists(csv_std_1_path):
                overall_max_step = max(pd.read_csv(csv_mean_0_path)['Step'].max(), pd.read_csv(csv_mean_1_path)['Step'].max())
                steps_per_epoch_corrected = overall_max_step / dataset_info['epochs']
                process_and_plot_combined(csv_mean_0_path, csv_mean_1_path, csv_std_0_path, csv_std_1_path,
                                          steps_per_epoch_corrected, dataset_info['xlim'], axs[idx], dataset_name,
                                          first=first, left=idx % 2==0, bottom=n - idx < 3)

            print(f'Plotted {dataset_name} {idx}.')
            first= False

        output_path = os.path.join(entropy_plots_dir, f"combined-entropy-plot-{fig_num + 1}.pdf" if all else 'products-and-dblp.pdf')
        plt.tight_layout()
        plt.savefig(output_path, format='pdf')

        # plt.show()

datasets_info = {
    'Citeseer': {'epochs': 50, 'xlim': 50},
    'Cora': {'epochs': 50, 'xlim': 50},
    'Flickr': {'epochs': 100, 'xlim': 100},
    'ogbn-arxiv': {'epochs': 150, 'xlim': 150},
    'ogbn-products': {'epochs': 100, 'xlim': 100},
    'Pubmed': {'epochs': 100, 'xlim': 100},
    'Reddit': {'epochs': 50, 'xlim': 50},
    'Yelp': {'epochs': 100, 'xlim': 100},
    'BlogCat': {'epochs': 100, 'xlim': 100},
    'snap-patents': {'epochs': 100, 'xlim': 100},
    'DBLP': {'epochs': 100, 'xlim': 100},
    'ogbn-proteins': {'epochs': 150, 'xlim': 150},
}

if __name__ == "__main__":
    entropy_plots_dir = './entropy_plots'

    generate_combined_plot_single(datasets_info, entropy_plots_dir)
    generate_combined_plot_single(datasets_info, entropy_plots_dir, all=False)


# plot_single_dataset('ogbn-proteins', datasets_info, entropy_plots_dir)

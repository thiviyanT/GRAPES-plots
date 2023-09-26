import pandas as pd
import numpy as np


def compute_confidence_intervals(df):
    """
    Computes the 95% confidence intervals based on the mean and standard deviation in the dataframe.
    """
    # Computing the standard error
    df['Standard Error'] = (df['Accuracy(std)'].astype(float) / np.sqrt(df['Sampling Number'].replace('All', np.nan).astype(float))).round(2)
    
    # Computing the 95% confidence interval
    z_value = 1.96  # Assuming normal distribution
    df['CI Lower'] = (df['Accuracy(mean)'].astype(float) - z_value * df['Standard Error']).round(2)
    df['CI Upper'] = (df['Accuracy(mean)'].astype(float) + z_value * df['Standard Error']).round(2)
    
    return df


dataframe = pd.read_csv("./results-uncertainty.csv")
dataframe_with_ci = compute_confidence_intervals(dataframe)

# Exporting the dataframe with confidence intervals to a CSV file
output_path = "./results_with_confidence_intervals.csv"
dataframe_with_ci.to_csv(output_path, index=False)


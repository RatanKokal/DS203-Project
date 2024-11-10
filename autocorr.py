import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Function to extract the first MFCC and calculate ACF and PACF
def process_mfcc_file(filename):
    # Read the CSV file (MFCCs)
    mfcc_df = pd.read_csv(filename, header=None)
    
    # Transpose the dataframe to get features along the columns
    mfcc_df = mfcc_df.transpose()
    
    # Extract the first MFCC (first column)
    mfcc_first = mfcc_df.values[:, 0]
    
    # Plot ACF (Autocorrelation)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plot_acf(mfcc_first, lags=250, ax=plt.gca())  # Adjust lags as needed
    plt.title(f"Autocorrelation (ACF) - {filename}")

    # Plot PACF (Partial Autocorrelation)
    plt.subplot(1, 2, 2)
    plot_pacf(mfcc_first, lags=250, ax=plt.gca())  # Adjust lags as needed
    plt.title(f"Partial Autocorrelation (PACF) - {filename}")

    # Save the plots as images (optional)
    plot_filename = f"{os.path.splitext(filename)[0]}_acf_pacf.png"
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=100)
    plt.show()

    print(f"ACF and PACF for {filename} saved as {plot_filename}")

# Get list of CSV files in the current directory
csv_files = [f for f in os.listdir('.') if f.endswith('_MFCC.csv')]

# Process each file
for file in csv_files:
    process_mfcc_file(file)

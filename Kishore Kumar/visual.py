import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import os

# Get the list of files that match the pattern '-MFCC.csv'
files = [f for f in os.listdir('.') if f.endswith('_MFCC.csv')]

# Set up a figure with smaller subplots for each file
fig, axs = plt.subplots(len(files), 1, figsize=(8, len(files) * 2))  # Adjust the figsize for better fit

# Iterate through each file and plot only the moving averages for the 1st and 2nd MFCC
for idx, file in enumerate(files):
    filename = os.path.join('.', file)
    # Load MFCCs from CSV
    mfcc_df = pd.read_csv(filename, header=None)

    # Transpose to get the correct orientation
    mfcc_df = mfcc_df.transpose()

    # Extract the first three MFCCs
    mfcc_first_three = mfcc_df.values[:, 3:6]

    # Calculate the moving average (window = 100) for 1st and 2nd MFCC
    moving_avg_mfcc_1 = pd.Series(mfcc_first_three[:, 0]).rolling(window=100).mean()
    moving_avg_mfcc_2 = pd.Series(mfcc_first_three[:, 1]).rolling(window=100).mean()

    # Plot only the moving averages for 1st and 2nd MFCC
    axs[idx].plot(moving_avg_mfcc_1, label='Moving Avg 1st MFCC', linestyle='--', color='black')
    axs[idx].plot(moving_avg_mfcc_2, label='Moving Avg 2nd MFCC', linestyle='--', color='green')

    # Set the title for each subplot
    axs[idx].set_title(f'File {file}', fontsize=6)  # Smaller title font

    # Add legend to the first subplot only to avoid clutter
    if idx == 0:
        axs[idx].legend(loc='upper right', fontsize='x-small')

    # Set limits and labels for each subplot
    axs[idx].set_xlim([0, mfcc_first_three.shape[0]])  # Set x-axis limits based on number of frames
    axs[idx].set_xlabel('Frame Index', fontsize=4)  # Smaller label font
    axs[idx].set_ylabel('Value', fontsize=4)

# Adjust layout to minimize spacing between subplots
plt.tight_layout()

# Save the final figure (optional)
plt.savefig('moving_avg_mfcc1_mfcc2.png', dpi=100)

# Show the plot
plt.show()

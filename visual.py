import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import os

# Get the list of files that match the pattern '_MFCC.csv'
files = [f for f in os.listdir('.') if f.endswith('_MFCC.csv')]

# Set up a figure with smaller subplots for each file (115 files)
fig, axs = plt.subplots(len(files), 1, figsize=(8, len(files) * 2))  # Adjust the figsize for better fit

# Iterate through each file and plot the features
for idx, file in enumerate(files):
    filename = os.path.join('.', file)
    # Load MFCCs from CSV
    mfcc_df = pd.read_csv(filename, header=None)

    mfcc_df = mfcc_df.transpose()

    # Extract the first three MFCCs
    mfcc_first_three = mfcc_df.values[:, :3]

    # Calculate delta and delta-delta of the first MFCC
    delta_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0])  # Delta of the 1st MFCC
    delta2_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0], order=2)  # Delta-delta of the 1st MFCC

    # Calculate the moving average (window = 100) for 1st and 2nd MFCC
    moving_avg_mfcc_1 = pd.Series(mfcc_first_three[:, 0]).rolling(window=100).mean()
    moving_avg_mfcc_2 = pd.Series(mfcc_first_three[:, 1]).rolling(window=100).mean()

    # Plot all features in the same subplot for the current file
    axs[idx].plot(mfcc_first_three[:, 0], label='1st MFCC')  # 1st MFCC
    axs[idx].plot(mfcc_first_three[:, 1], label='2nd MFCC')  # 2nd MFCC
    axs[idx].plot(mfcc_first_three[:, 2], label='3rd MFCC')  # 3rd MFCC
    axs[idx].plot(delta_mfcc_1, label='Delta 1st MFCC')      # Delta 1st MFCC
    axs[idx].plot(delta2_mfcc_1, label='Delta-Delta 1st MFCC')  # Delta-Delta 1st MFCC

    # Plot moving averages for 1st and 2nd MFCC
    axs[idx].plot(moving_avg_mfcc_1, label='Moving Avg 1st MFCC', linestyle='--', color='black')  # Moving avg of 1st MFCC
    axs[idx].plot(moving_avg_mfcc_2, label='Moving Avg 2nd MFCC', linestyle='--', color='green')  # Moving avg of 2nd MFCC

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
plt.savefig('all_files_smaller_features_with_moving_avg.png', dpi=100)

# Show the plot
plt.show()

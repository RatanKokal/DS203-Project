import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# Set up a figure with smaller subplots for 115 files (adjust figsize to fit the plots)
fig, axs = plt.subplots(115, 1, figsize=(8, 150))  # Smaller height for each subplot (was 10,5000px, now 8,150)

for file in range(1, 116):
    if file < 10:
        filename = "0" + str(file) + "-MFCC.csv"
    else:
        filename = str(file) + "-MFCC.csv"
    
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
    axs[file-1].plot(mfcc_first_three[:, 0], label='1st MFCC')  # 1st MFCC
    axs[file-1].plot(mfcc_first_three[:, 1], label='2nd MFCC')  # 2nd MFCC
    axs[file-1].plot(mfcc_first_three[:, 2], label='3rd MFCC')  # 3rd MFCC
    axs[file-1].plot(delta_mfcc_1, label='Delta 1st MFCC')      # Delta 1st MFCC
    axs[file-1].plot(delta2_mfcc_1, label='Delta-Delta 1st MFCC')  # Delta-Delta 1st MFCC

    # Plot moving averages for 1st and 2nd MFCC
    axs[file-1].plot(moving_avg_mfcc_1, label='Moving Avg 1st MFCC', linestyle='--', color='black')  # Moving avg of 1st MFCC
    axs[file-1].plot(moving_avg_mfcc_2, label='Moving Avg 2nd MFCC', linestyle='--', color='green')  # Moving avg of 2nd MFCC
    
    # Set the title for each subplot
    axs[file-1].set_title(f'File {file}', fontsize=6)  # Smaller title font
    
    # Only add legend to the first subplot (to avoid clutter)
    if file == 1:
        axs[file-1].legend(loc='upper right', fontsize='x-small')

    # Set limits and labels for each subplot
    axs[file-1].set_xlim([0, mfcc_first_three.shape[0]])  # Set x-axis limits based on number of frames
    axs[file-1].set_xlabel('Frame Index', fontsize=4)  # Smaller label font
    axs[file-1].set_ylabel('Value', fontsize=4)

# Adjust layout to minimize spacing between subplots
plt.tight_layout()

# Save the final figure (optional)
plt.savefig('all_files_smaller_features_with_moving_avg.png', dpi=100)

# Show the plot

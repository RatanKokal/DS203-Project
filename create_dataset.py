import os
import pandas as pd
import librosa
import numpy as np
from scipy.stats import skew, kurtosis

def calculate_additional_features(data, window_size=30):
    """Calculate additional features averaged over a moving window."""
    # Peak to peak spread in windows
    peak_to_peak_spread = []
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size]
        peak_to_peak_spread.append(np.max(window) - np.min(window))
    avg_peak_to_peak = np.mean(peak_to_peak_spread)
    
    # RMS (Root Mean Square)
    rms = np.sqrt(np.mean(np.square(data)))
    
    # Spectral Centroid
    freqs = np.fft.fftfreq(len(data))
    spec = np.abs(np.fft.fft(data))
    spectral_centroid = np.sum(freqs * spec) / np.sum(spec)
    
    # Range value (max - min)
    range_val = np.max(data) - np.min(data)
    
    # Max and min values
    max_val = np.max(data)
    min_val = np.min(data)
    
    return {
        'avg_peak_to_peak': avg_peak_to_peak,
        'rms': rms,
        'spectral_centroid': spectral_centroid,
        'range_val': range_val,
        'max_val': max_val,
        'min_val': min_val
    }

def calculate_max_delta_change(data, slab_size=105, window_size=15):
    """Calculate the sum of max - min for continuously rising or falling sequences in slabs."""
    # Initialize the list to store max - min of continuously rising/falling arrays
    delta_changes = []
    
    # Loop through data in slabs
    for slab_start in range(0, len(data) - slab_size + 1, slab_size):
        slab = data[slab_start:slab_start + slab_size]

        # Now process the slab in windows of 15
        for i in range(0, len(slab) - window_size + 1, window_size):
            window = slab[i:i + window_size]

            # Identify continuously rising or falling subarrays
            rising = [window[0]]
            falling = [window[0]]
            
            for j in range(1, len(window)):
                if window[j] >= window[j-1]:  # Check for rising
                    rising.append(window[j])
                else:
                    if len(rising) > 1:  # Process the rising subarray
                        delta_changes.append(max(rising) - min(rising))
                    rising = [window[j]]  # Reset for new sequence
                
                if window[j] <= window[j-1]:  # Check for falling
                    falling.append(window[j])
                else:
                    if len(falling) > 1:  # Process the falling subarray
                        delta_changes.append(max(falling) - min(falling))
                    falling = [window[j]]  # Reset for new sequence

            # At the end of the window, check any remaining sequence
            if len(rising) > 1:
                delta_changes.append(max(rising) - min(rising))
            if len(falling) > 1:
                delta_changes.append(max(falling) - min(falling))

    # Sum of all max - min differences divided by total number of frames
    total_frames = len(data)
    if total_frames > 0:
        return np.sum(delta_changes) / total_frames
    else:
        return 0  # If there are no frames, return 0

def calculate_moving_average_abs_diff(data, window_size=100):
    """Calculate the sum of absolute differences between moving average and actual values."""
    moving_average = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    # Adjust data length to match moving average length
    actual_values = data[window_size - 1:]
    
    abs_diff = np.abs(moving_average - actual_values)
    sum_abs_diff = np.sum(abs_diff) / len(data)
    
    return sum_abs_diff

def create_dataset_from_all_folders(parent_folder, window_size=30):
    # Initialize an empty list to hold the feature vectors
    features_list = []

    # Loop through all subfolders in the specified parent folder
    class_label = 1  # Start class label from 1
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path) and folder_name != '.git' and folder_name != 'Testing':
            print(f"Processing folder: {folder_name} (Class {class_label})")
            
            # Loop through all CSV files in the current folder
            for filename in os.listdir(folder_path):
                if filename.endswith('_MFCC.csv'):
                    file_path = os.path.join(folder_path, filename)
                    
                    # Load MFCCs from CSV
                    mfcc_df = pd.read_csv(file_path, header=None)
                    
                    # Transpose the DataFrame to ensure the rows represent the frames and columns represent the features
                    mfcc_df = mfcc_df.transpose()
                    
                    # Extract all 20 MFCCs
                    mfcc_all = mfcc_df.values[:, :20]
                    
                    # Initialize lists to store the computed features for all 20 MFCCs
                    feature_means = []
                    feature_stds = []
                    rms_features = []
                    skew_features = []
                    kurtosis_features = []
                    additional_feature_list = []  # To hold additional features for 1st and 2nd MFCCs

                    # Loop through the first 20 MFCCs
                    for i in range(20):
                        # Calculate delta of each MFCC
                        delta_mfcc = librosa.feature.delta(mfcc_all[:, i])  # Delta of current MFCC
                        
                        # Compute the mean, std, RMS, skewness, kurtosis, and delta for the current MFCC
                        feature_means.append(np.mean(mfcc_all[:, i]))
                        feature_stds.append(np.std(mfcc_all[:, i]))
                        rms_features.append(np.sqrt(np.mean(np.square(mfcc_all[:, i]))))  # RMS of the current MFCC
                        skew_features.append(skew(mfcc_all[:, i]))  # Skewness of the current MFCC
                        kurtosis_features.append(kurtosis(mfcc_all[:, i]))  # Kurtosis of the current MFCC

                    # Calculate additional features for the 1st and 2nd MFCCs
                    additional_features_1st = calculate_additional_features(mfcc_all[:, 0], window_size)
                    additional_features_2nd = calculate_additional_features(mfcc_all[:, 1], window_size)
                    
                    # Calculate maximum delta change for MFCC 1 and MFCC 2 using slabs of 105 frames
                    max_delta_mfcc_1 = calculate_max_delta_change(mfcc_all[:, 0], slab_size=105, window_size=15)
                    max_delta_mfcc_2 = calculate_max_delta_change(mfcc_all[:, 1], slab_size=105, window_size=15)
                    
                    # Calculate sum of absolute differences between moving average and actual values
                    moving_avg_abs_diff_1st = calculate_moving_average_abs_diff(mfcc_all[:, 0], window_size=100)
                    moving_avg_abs_diff_2nd = calculate_moving_average_abs_diff(mfcc_all[:, 1], window_size=100)
                    
                    # Combine additional features for the 1st and 2nd MFCCs
                    additional_feature_list.extend([
                        additional_features_1st['avg_peak_to_peak'],
                        additional_features_1st['rms'],
                        additional_features_1st['spectral_centroid'],
                        additional_features_1st['range_val'],
                        additional_features_1st['max_val'],
                        additional_features_1st['min_val'],
                        additional_features_2nd['avg_peak_to_peak'],
                        additional_features_2nd['rms'],
                        additional_features_2nd['spectral_centroid'],
                        additional_features_2nd['range_val'],
                        additional_features_2nd['max_val'],
                        additional_features_2nd['min_val'],
                        max_delta_mfcc_1,  # Max delta change for MFCC 1
                        max_delta_mfcc_2,  # Max delta change for MFCC 2
                        moving_avg_abs_diff_1st,  # Sum of abs diff for MFCC 1
                        moving_avg_abs_diff_2nd   # Sum of abs diff for MFCC 2
                    ])
                    
                    # Combine all features (mean, std, RMS, skewness, kurtosis, additional features) into one list
                    file_features = (feature_means + feature_stds + rms_features + 
                                     skew_features + kurtosis_features + additional_feature_list)
                    
                    # Add the class label to the features list
                    file_features.append(class_label)
                    
                    # Append the features and label for this file to the list
                    features_list.append(file_features)

            # Increment class label for the next folder
            class_label += 1

    # Define column names
    columns = [
        *[f'mean_mfcc{i+1}' for i in range(20)],
        *[f'std_mfcc{i+1}' for i in range(20)],
        *[f'rms_mfcc{i+1}' for i in range(20)],
        *[f'skew_mfcc{i+1}' for i in range(20)],
        *[f'kurtosis_mfcc{i+1}' for i in range(20)],
        'avg_peak_to_peak_1st_MFCC', 'RMS_1st_MFCC', 'Spectral_Centroid_1st_MFCC', 
        'Range_Val_1st_MFCC', 'Max_Val_1st_MFCC', 'Min_Val_1st_MFCC',
        'avg_peak_to_peak_2nd_MFCC', 'RMS_2nd_MFCC', 'Spectral_Centroid_2nd_MFCC', 
        'Range_Val_2nd_MFCC', 'Max_Val_2nd_MFCC', 'Min_Val_2nd_MFCC',
        'Max_Delta_Change_1st_MFCC', 'Max_Delta_Change_2nd_MFCC',  # Updated features
        'Sum_Abs_Diff_Moving_Avg_1st_MFCC', 'Sum_Abs_Diff_Moving_Avg_2nd_MFCC',  # New features
        'Class'
    ]
    
    # Convert the list of features to a DataFrame
    features_df = pd.DataFrame(features_list, columns=columns)
    
    # Save the features DataFrame to a CSV file
    output_file = os.path.join(parent_folder, 'database_updated.csv')
    features_df.to_csv(output_file, index=False)
    
    print(f"Combined dataset saved to {output_file}")

# Example usage
parent_folder = '.'  # Replace with the path to your parent folder containing subfolders
create_dataset_from_all_folders(parent_folder, window_size=30)

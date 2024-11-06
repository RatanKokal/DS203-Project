import os
import pandas as pd
import librosa
import numpy as np
from scipy.stats import skew, kurtosis

def create_dataset_from_all_folders(parent_folder):
    # Initialize an empty list to hold the feature vectors
    features_list = []

    # Loop through all subfolders in the specified parent folder
    class_label = 1  # Start class label from 1
    for folder_name in os.listdir(parent_folder):
        folder_path = os.path.join(parent_folder, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
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
                    delta_features = []

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
                        # delta_features.append(np.mean(delta_mfcc))  # Mean of delta of the current MFCC
                    
                    # Combine all features (mean, std, RMS, skewness, kurtosis, delta) into one list
                    file_features = (feature_means + feature_stds + rms_features + 
                                     skew_features + kurtosis_features)
                    
                    # Add the class label to the features list
                    file_features.append(class_label)
                    
                    # Append the features and label for this file to the list
                    features_list.append(file_features)

            # Increment class label for the next folder
            class_label += 1

    # Convert the list of features to a DataFrame
    features_df = pd.DataFrame(features_list, columns=[
        *[f'mean_mfcc{i+1}' for i in range(20)],
        *[f'std_mfcc{i+1}' for i in range(20)],
        *[f'rms_mfcc{i+1}' for i in range(20)],
        *[f'skew_mfcc{i+1}' for i in range(20)],
        *[f'kurtosis_mfcc{i+1}' for i in range(20)],
        # *[f'mean_delta_mfcc{i+1}' for i in range(20)],
        'Class'
    ])
    
    # Save the features DataFrame to a CSV file
    output_file = os.path.join(parent_folder, 'database.csv')
    features_df.to_csv(output_file, index=False)
    
    print(f"Combined dataset saved to {output_file}")

# Example usage
parent_folder = '.'  # Replace with the path to your parent folder containing subfolders
create_dataset_from_all_folders(parent_folder)

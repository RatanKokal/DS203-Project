import os
import pandas as pd
import librosa
import numpy as np

def create_dataset_from_csv(folder_path):
    # Initialize an empty list to hold the feature vectors
    features_list = []

    # Ask user for the class label for this file
    class_label = input(f"Enter class label")

    # Loop through all CSV files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('_MFCC.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # Load MFCCs from CSV
            mfcc_df = pd.read_csv(file_path, header=None)
            
            # Transpose the DataFrame to ensure the rows represent the frames and columns represent the features
            mfcc_df = mfcc_df.transpose()
            
            # Extract the first three MFCCs
            mfcc_first_three = mfcc_df.values[:, :3]
            
            # Calculate delta and delta-delta of the first MFCC
            delta_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0])  # Delta of 1st MFCC
            delta2_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0], order=2)  # Delta-Delta of 1st MFCC
            
            # Compute the mean and standard deviation for the MFCC and delta/delta-delta features
            feature_means = [
                np.mean(mfcc_first_three[:, 0]),  # Mean of 1st MFCC
                np.mean(mfcc_first_three[:, 1]),  # Mean of 2nd MFCC
                np.mean(mfcc_first_three[:, 2]),  # Mean of 3rd MFCC
                np.mean(delta_mfcc_1),            # Mean of delta of 1st MFCC
                np.mean(delta2_mfcc_1)            # Mean of delta-delta of 1st MFCC
            ]
            
            feature_stds = [
                np.std(mfcc_first_three[:, 0]),   # Std dev of 1st MFCC
                np.std(mfcc_first_three[:, 1]),   # Std dev of 2nd MFCC
                np.std(mfcc_first_three[:, 2]),   # Std dev of 3rd MFCC
                np.std(delta_mfcc_1),             # Std dev of delta of 1st MFCC
                np.std(delta2_mfcc_1)             # Std dev of delta-delta of 1st MFCC
            ]
            
            # Combine the mean and std features into one list (total 10 features per file)
            file_features = feature_means + feature_stds
            
            # Add the class label to the features list
            file_features.append(class_label)
            
            # Append the features and label for this file to the list
            features_list.append(file_features)

    # Convert the list of features to a DataFrame
    features_df = pd.DataFrame(features_list, columns=[
        'mean_mfcc1', 'mean_mfcc2', 'mean_mfcc3', 'mean_delta_mfcc1', 'mean_delta2_mfcc1',
        'std_mfcc1', 'std_mfcc2', 'std_mfcc3', 'std_delta_mfcc1', 'std_delta2_mfcc1', 'Class'
    ])
    
    # Save the features DataFrame to a CSV file
    output_file = os.path.join(folder_path, 'database.csv')
    features_df.to_csv(output_file, index=False)
    
    print(f"Combined dataset saved to {output_file}")

# Example usage
folder_path = '.'  # Replace with the path to your folder containing the CSV files
create_dataset_from_csv(folder_path)

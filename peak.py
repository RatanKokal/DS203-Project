import os
import pandas as pd
import numpy as np

# Function to compute the average peak-to-peak spread within a window
def avg_peak_to_peak_spread(data, window_size=10):
    peak_to_peak_spread = []
    
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size]
        peak_to_peak_spread.append(np.max(window) - np.min(window))
    
    # Return the sum of all peak-to-peak values divided by the length of the MFCC
    return np.sum(peak_to_peak_spread) / len(data)

def calculate_peak_to_peak_spread_of_mfccs(folder_path, output_csv='peak_to_peak_spread_results.csv'):
    # List to store the results
    results = []

    # Loop through all CSV files in the specified folder
    for filename in os.listdir(folder_path):
        if filename.endswith('-MFCC.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # Load MFCCs from CSV
            mfcc_df = pd.read_csv(file_path, header=None)
            
            # Transpose the DataFrame to ensure the rows represent the frames and columns represent the features
            mfcc_df = mfcc_df.transpose()
            
            # Extract the 1st and 2nd MFCCs (1st at index 0, 2nd at index 1)
            mfcc_1st = mfcc_df.values[:, 0]
            mfcc_2nd = mfcc_df.values[:, 1]
            
            # Calculate average peak-to-peak spread for both MFCC 1 and MFCC 2
            avg_peak_to_peak_1st_mfcc = avg_peak_to_peak_spread(mfcc_1st)
            avg_peak_to_peak_2nd_mfcc = avg_peak_to_peak_spread(mfcc_2nd)
            
            # Append the results to the list
            results.append([filename, avg_peak_to_peak_1st_mfcc, avg_peak_to_peak_2nd_mfcc])

    # Convert results to a DataFrame and save to CSV
    results_df = pd.DataFrame(results, columns=['Filename', 'Avg_Peak_to_Peak_1st_MFCC', 'Avg_Peak_to_Peak_2nd_MFCC'])
    output_file = os.path.join(folder_path, output_csv)
    results_df.to_csv(output_file, index=False)
    
    print(f"Results saved to {output_file}")

# Example usage
folder_path = '.'  # Current directory
calculate_peak_to_peak_spread_of_mfccs(folder_path)

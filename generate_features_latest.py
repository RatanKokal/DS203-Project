import os
import pandas as pd
import numpy as np

def calculate_features(data, window_size=100):
    """Calculate various features averaged over a moving window."""
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

def calculate_mfcc_features(folder_path, output_csv='mfcc_features_results.csv', window_size=100):
    results = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith('-MFCC.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # Load MFCCs from CSV
            mfcc_df = pd.read_csv(file_path, header=None)
            mfcc_df = mfcc_df.transpose()
            
            # Extract 1st and 2nd MFCCs
            mfcc_1st = mfcc_df.values[:, 0]
            mfcc_2nd = mfcc_df.values[:, 1]
            
            # Calculate features for both MFCCs with moving window
            features_1st = calculate_features(mfcc_1st, window_size)
            features_2nd = calculate_features(mfcc_2nd, window_size)
            
            # Combine results
            result_row = [
                filename,
                # 1st MFCC features
                features_1st['avg_peak_to_peak'],
                features_1st['rms'],
                features_1st['spectral_centroid'],
                features_1st['range_val'],
                features_1st['max_val'],
                features_1st['min_val'],
                # 2nd MFCC features
                features_2nd['avg_peak_to_peak'],
                features_2nd['rms'],
                features_2nd['spectral_centroid'],
                features_2nd['range_val'],
                features_2nd['max_val'],
                features_2nd['min_val']
            ]
            
            results.append(result_row)
    
    # Create DataFrame with labeled columns
    columns = ['Filename']
    for mfcc_num in ['1st', '2nd']:
        for feature in ['Avg_Peak_to_Peak', 'RMS', 'Spectral_Centroid', 'Range_Val', 'Max_Val', 'Min_Val']:
            columns.append(f'{feature}_{mfcc_num}_MFCC')
    
    results_df = pd.DataFrame(results, columns=columns)
    
    # Save to CSV
    output_file = os.path.join(folder_path, output_csv)
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    return results_df

# Example usage
if __name__ == "__main__":
    folder_path = '.'  # Current directory
    window_size = 100  # Define the window size
    results = calculate_mfcc_features(folder_path, window_size=window_size)
    print("\nFirst few rows of results:")
    print(results.head())

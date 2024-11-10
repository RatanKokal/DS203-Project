import os
import pandas as pd
import joblib
import numpy as np
from scipy.stats import skew, kurtosis
import librosa

# Load the trained Random Forest model and scaler
rf_classifier = joblib.load('svm_(rbf_kernel)_model.pkl')  # Trained Random Forest model
scaler = joblib.load('scaler.pkl')  # Saved scaler for normalization

def calculate_additional_features(data, window_size=30):
    """Calculate additional features averaged over a moving window."""
    peak_to_peak_spread = []
    for i in range(0, len(data) - window_size + 1, window_size):
        window = data[i:i + window_size]
        peak_to_peak_spread.append(np.max(window) - np.min(window))
    avg_peak_to_peak = np.mean(peak_to_peak_spread)
    
    rms = np.sqrt(np.mean(np.square(data)))
    freqs = np.fft.fftfreq(len(data))
    spec = np.abs(np.fft.fft(data))
    spectral_centroid = np.sum(freqs * spec) / np.sum(spec)
    
    range_val = np.max(data) - np.min(data)
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
    delta_changes = []
    for slab_start in range(0, len(data) - slab_size + 1, slab_size):
        slab = data[slab_start:slab_start + slab_size]
        for i in range(0, len(slab) - window_size + 1, window_size):
            window = slab[i:i + window_size]
            rising, falling = [window[0]], [window[0]]
            for j in range(1, len(window)):
                if window[j] >= window[j-1]: rising.append(window[j])
                else:
                    if len(rising) > 1: delta_changes.append(max(rising) - min(rising))
                    rising = [window[j]]
                if window[j] <= window[j-1]: falling.append(window[j])
                else:
                    if len(falling) > 1: delta_changes.append(max(falling) - min(falling))
                    falling = [window[j]]
            if len(rising) > 1: delta_changes.append(max(rising) - min(rising))
            if len(falling) > 1: delta_changes.append(max(falling) - min(falling))
    total_frames = len(data)
    return np.sum(delta_changes) / total_frames if total_frames > 0 else 0

def calculate_moving_average_abs_diff(data, window_size=100):
    """Calculate the sum of absolute differences between moving average and actual values."""
    moving_average = np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    actual_values = data[window_size - 1:]
    abs_diff = np.abs(moving_average - actual_values)
    return np.sum(abs_diff) / len(data)

def classify_new_data(folder_path, window_size=30, output_file='classification_results.csv'):
    classification_results = []

    # Loop through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('-MFCC.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # Load MFCC features
            mfcc_df = pd.read_csv(file_path, header=None)
            mfcc_df = mfcc_df.transpose()
            mfcc_values = mfcc_df.values[:, :20]  # First 20 MFCCs
            
            feature_means, feature_stds, rms_features, skew_features, kurtosis_features = [], [], [], [], []
            additional_features = []

            for i in range(20):
                delta_mfcc = librosa.feature.delta(mfcc_values[:, i])
                feature_means.append(np.mean(mfcc_values[:, i]))
                feature_stds.append(np.std(mfcc_values[:, i]))
                rms_features.append(np.sqrt(np.mean(np.square(mfcc_values[:, i]))))
                skew_features.append(skew(mfcc_values[:, i]))
                kurtosis_features.append(kurtosis(mfcc_values[:, i]))
            
            # Additional features for 1st and 2nd MFCCs
            additional_features_1st = calculate_additional_features(mfcc_values[:, 0], window_size)
            additional_features_2nd = calculate_additional_features(mfcc_values[:, 1], window_size)
            
            max_delta_mfcc_1 = calculate_max_delta_change(mfcc_values[:, 0], slab_size=105, window_size=15)
            max_delta_mfcc_2 = calculate_max_delta_change(mfcc_values[:, 1], slab_size=105, window_size=15)
            
            moving_avg_abs_diff_1st = calculate_moving_average_abs_diff(mfcc_values[:, 0], window_size=100)
            moving_avg_abs_diff_2nd = calculate_moving_average_abs_diff(mfcc_values[:, 1], window_size=100)
            
            additional_features.extend([
                additional_features_1st['avg_peak_to_peak'], additional_features_1st['rms'], 
                additional_features_1st['spectral_centroid'], additional_features_1st['range_val'], 
                additional_features_1st['max_val'], additional_features_1st['min_val'], 
                additional_features_2nd['avg_peak_to_peak'], additional_features_2nd['rms'], 
                additional_features_2nd['spectral_centroid'], additional_features_2nd['range_val'], 
                additional_features_2nd['max_val'], additional_features_2nd['min_val'],
                max_delta_mfcc_1, max_delta_mfcc_2, moving_avg_abs_diff_1st, moving_avg_abs_diff_2nd
            ])
            
            # Combine all features into one list
            file_features = (feature_means + feature_stds + rms_features + 
                             skew_features + kurtosis_features + additional_features)

            # Scale features
            file_features_scaled = scaler.transform([file_features])
            
            # Classify with Random Forest
            predicted_class = rf_classifier.predict(file_features_scaled)
            classification_results.append((filename, predicted_class[0]))

    # Save results to CSV
    results_df = pd.DataFrame(classification_results, columns=['Filename', 'Predicted Class'])
    results_df.to_csv(output_file, index=False)
    print(f"Classification results saved to {output_file}")

# Example usage
folder_path = '.'  # Replace with your test folder path
output_file = 'classification_results.csv'  # Specify the output file
classify_new_data(folder_path, window_size=30, output_file=output_file)

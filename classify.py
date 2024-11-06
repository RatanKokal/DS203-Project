import os
import pandas as pd
import joblib
import numpy as np
from scipy.stats import skew, kurtosis
import librosa

# Step 1: Load the trained model and scaler
svm_classifier = joblib.load('svm_model.pkl')  # Load the trained SVM model
scaler = joblib.load('scaler.pkl')  # Load the saved scaler

# Step 2: Define a function to classify new data
def classify_new_data(folder_path):
    classification_results = []

    # Loop through all CSV files in the current folder
    for filename in os.listdir(folder_path):
        if filename.endswith('-MFCC.csv'):
            file_path = os.path.join(folder_path, filename)
            
            # Load MFCC features from CSV
            mfcc_df = pd.read_csv(file_path, header=None)
            mfcc_df = mfcc_df.transpose()  # Ensure rows represent frames, columns represent features
            mfcc_values = mfcc_df.values[:, :20]  # Only take the first 20 MFCCs
            
            # Compute features for classification
            feature_means = np.mean(mfcc_values, axis=0)
            feature_stds = np.std(mfcc_values, axis=0)
            rms_features = np.sqrt(np.mean(np.square(mfcc_values), axis=0))
            skew_features = skew(mfcc_values, axis=0)
            kurtosis_features = kurtosis(mfcc_values, axis=0)
            delta_features = np.mean(librosa.feature.delta(mfcc_values, axis=0), axis=0)
            
            # Combine features into a single array (in the specified order)
            features = np.hstack((
                feature_means, 
                feature_stds, 
                rms_features, 
                skew_features, 
                kurtosis_features, 
                delta_features
            ))
            
            # Standardize the features using the previously saved scaler
            features_scaled = scaler.transform([features])  # Transform new data
            
            # Classify the data using the trained SVM model
            predicted_class = svm_classifier.predict(features_scaled)[0]
            
            # Save the classification result (filename and predicted class)
            classification_results.append([filename, predicted_class])
    
    # Save the classification results to a CSV file
    results_df = pd.DataFrame(classification_results, columns=['Filename', 'Predicted Class'])
    output_file = os.path.join(folder_path, 'classification_results.csv')
    results_df.to_csv(output_file, index=False)
    
    print(f"Classification results saved to {output_file}")

# Step 3: Classify all files in the current folder
folder_path = '.'  # Replace with your folder containing new CSV files
classify_new_data(folder_path)

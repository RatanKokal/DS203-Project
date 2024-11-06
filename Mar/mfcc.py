# Uncomment the following line if you do not have the Python module 'librosa' installed
# !pip install librosa

import os
import numpy as np
import pandas as pd
import librosa

# Function to create MFCC coefficients given an audio file
def create_MFCC_coefficients(file_name):
    sr_value = 44100  # Sample rate
    n_mfcc_count = 20  # Number of MFCCs

    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_name, sr=sr_value)
        
        # Compute MFCC coefficients for the audio
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_count)

        # Create a DataFrame from the MFCCs
        coeff_df = pd.DataFrame(mfccs)

        return coeff_df

    except Exception as e:
        print(f"Error creating MFCC coefficients for {file_name}: {str(e)}")
        return None

# Get the current working directory
current_directory = os.getcwd()

# Loop through all MP3 files in the directory
for filename in os.listdir(current_directory):
    if filename.endswith('.mp3'):
        filepath = os.path.join(current_directory, filename)
        print(f"Processing {filename}...")

        # Create MFCCs and save them to CSV
        mfcc_df = create_MFCC_coefficients(filepath)
        if mfcc_df is not None:
            csv_filename = os.path.splitext(filename)[0] + '_MFCC.csv'
            mfcc_df.to_csv(csv_filename, index=False, header=False)
            print(f"MFCC coefficients saved to {csv_filename}")

import os
import pandas as pd
import librosa

# Function to create MFCC coefficients given an audio file
def create_MFCC_coefficients(file_name):
    sr_value = 44100  # Sample rate
    n_mfcc_count = 20  # Number of MFCCs

    try:
        # Load the MP3 file directly using librosa
        y, sr = librosa.load(file_name, sr=sr_value)

        # Compute MFCC coefficients
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_count)

        # Create a DataFrame from the MFCCs
        mfcc_df = pd.DataFrame(mfccs)

        # Save the MFCC coefficients to a CSV file
        csv_filename = os.path.splitext(file_name)[0] + '_MFCC.csv'
        mfcc_df.to_csv(csv_filename, index=False, header=False)
        print(f"MFCC coefficients saved to {csv_filename}")

    except Exception as e:
        print(f"Error creating MFCC coefficients for {file_name}: {str(e)}")

current_directory = os.getcwd()

# Loop through all MP3 files in the directory
for filename in os.listdir(current_directory):
    if filename.endswith('.mp3'):
        filepath = os.path.join(current_directory, filename)
        print(f"Processing {filename}...")
        
        # Create MFCCs and save them to CSV
        create_MFCC_coefficients(filepath)

import pandas as pd
import os
import glob

# Function to cut the first 1000 and last 1000 frames of the MFCC data, keeping only the middle frames
def crop_mfcc(file_path):
    # Load the CSV data
    mfcc_df = pd.read_csv(file_path, header=None)
    
    # Transpose if needed (optional, depending on your data structure)
    mfcc_df = mfcc_df.transpose()

    # Keep only the middle frames (excluding the first 1000 and last 1000 rows)
    cropped_df = mfcc_df.iloc[1000:-1000].reset_index(drop=True)

    # Transpose back if needed (depending on your original format)
    cropped_df = cropped_df.transpose()
    
    # Save the cropped data back to the same file
    cropped_df.to_csv(file_path, index=False, header=False)
    print(f"Cropped and saved: {file_path}")

# Get all the CSV files ending with "_MFCC.csv" in the current directory
mfcc_files = glob.glob('*_MFCC.csv')

# Iterate through each file and crop it
for mfcc_file in mfcc_files:
    crop_mfcc(mfcc_file)

print("Processing completed for all files.")

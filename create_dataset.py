import os
import pandas as pd

def merge_csv_from_subfolders(folder_path):
    # Initialize an empty list to store the dataframes
    all_dataframes = []

    # Walk through all subdirectories and files in the folder
    for root, dirs, files in os.walk(folder_path):
        # Check if 'database.csv' exists in the current directory
        if 'database.csv' in files:
            file_path = os.path.join(root, 'database.csv')
            print(file_path)
            # Load the CSV into a DataFrame
            df = pd.read_csv(file_path)
            # Append to the list of dataframes
            all_dataframes.append(df)

    # Concatenate all dataframes into one
    if all_dataframes:
        merged_df = pd.concat(all_dataframes, ignore_index=True)
        # Save the merged DataFrame to a new 'database.csv' in the current folder
        merged_df.to_csv('database.csv', index=False)
        print(f"All CSV files have been merged and saved to 'database.csv'.")
    else:
        print("No 'database.csv' files found in the subfolders.")

# Example usage
merge_csv_from_subfolders('.')  # Replace with the path to your main folder

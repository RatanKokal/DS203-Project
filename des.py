import pandas as pd
import matplotlib.pyplot as plt

# Load the MFCC data from a CSV file
mfcc_df = pd.read_csv('मल परतचय झलयत झलव - मरठ लवण गत  MALA PIRTICHYA - AIYAA  SUPERHIT Lyrical LAVNI SONG-[AudioTrimmer.com]_MFCC.csv', header=None)

# Transpose the data if needed
mfcc_df = mfcc_df.transpose()

# Plot all 20 parameters on a single graph
plt.figure(figsize=(14, 8))

# Loop through each of the 20 MFCC parameters and plot them
for i in range(20):
    plt.plot(mfcc_df[i][5000:5100], label=f'Parameter {i+1}')  # Select the range from index 5000 to 5100

# Adding labels and title
plt.title('Overlay of 20 MFCC Parameters', fontsize=16)
plt.xlabel('Sample Index', fontsize=12)
plt.ylabel('MFCC Value', fontsize=12)

# Display the legend
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))  # Adjust the legend location
plt.tight_layout()

# Show the plot
plt.show()

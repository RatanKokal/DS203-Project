import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the MFCC data from a CSV file
mfcc_df = pd.read_csv('01-MFCC.csv')

# Calculate the correlation matrix
# correlation_matrix = mfcc_df.corr()

# # Plot the correlation matrix as a heatmap
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt=".2f",
#             cbar_kws={"shrink": .8}, annot_kws={"size": 8})
# plt.title("MFCC Correlation Matrix")
# plt.xlabel("MFCC Coefficients")
# plt.ylabel("MFCC Coefficients")
# plt.show()
for index in range(1,117):
    if index < 10:
        filename = "0" + str(index) + "-MFCC.csv"
    else:
        filename = str(index) + "-MFCC.csv"
    mfcc_df = pd.read_csv(filename)
    if mfcc_df.shape[1] <= 12000:
        print(f"File {filename} satisfies the condition, It has shape {mfcc_df.shape}")

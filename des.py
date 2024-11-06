# # # import pandas as pd
# # # import seaborn as sns
# # # import matplotlib.pyplot as plt

# # # # Load the MFCC data from a CSV file
# # # mfcc_df = pd.read_csv('29-MFCC.csv', header=None)
# # # # mfcc_df.transpose().reset_index().head()
# # # # mfcc_df = mfcc_df.reset_index()
# # # # mfcc_df.head()
# # # # mfcc_df.columns = range(mfcc_df.shape[1])  # Temporarily give default numeric column names
# # # # mfcc_df = pd.concat([mfcc_df.iloc[0:1], mfcc_df])  # Add a duplicate of the first row to the top
# # # # mfcc_df = mfcc_df.reset_index(drop=True)
# # # mfcc_df = mfcc_df.transpose()
# # # print(mfcc_df.describe())
# # # # mfcc_cropped = mfcc_df.iloc[:,5000:5400]

# # # # Calculate the correlation matrix
# # # # correlation_matrix = mfcc_cropped.corr()

# # # # # # Plot the correlation matrix as a heatmap
# # # # plt.figure(figsize=(10, 8))
# # # # sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True, fmt=".2f",
# # # #             cbar_kws={"shrink": .8}, annot_kws={"size": 8})
# # # # plt.title("MFCC Correlation Matrix")
# # # # plt.xlabel("MFCC Coefficients")
# # # # plt.ylabel("MFCC Coefficients")
# # # # plt.savefig('mfcc_correlation.png')
# # # # plt.show()

# # import pandas as pd
# # import seaborn as sns
# # import matplotlib.pyplot as plt

# # # Load the MFCC data from a CSV file
# # mfcc_df = pd.read_csv('mj1-MFCC.csv', header=None)

# # # Transpose the data if needed
# # mfcc_df = mfcc_df.transpose()

# # # Plot all 20 parameters on a single graph
# # plt.figure(figsize=(14, 8))

# # for i in range(20):
# #     sns.lineplot(data=mfcc_df[i], label=f'Parameter {i+1}')  # Plot each parameter with a label

# # plt.title("MFCC Parameters Overlaid")
# # plt.xlabel("Sample Index")
# # plt.ylabel("MFCC Value")
# # plt.legend(loc="upper right", bbox_to_anchor=(1.15, 1))  # Adjust legend to fit outside the plot area
# # plt.tight_layout()  # Adjust layout to prevent clipping
# # plt.savefig('mfcc_all_parameters_overlay.png')
# # plt.show()

# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Load the MFCC data from a CSV file
# mfcc_df = pd.read_csv('09-MFCC.csv', header=None)

# # Transpose the data if needed
# mfcc_df = mfcc_df.transpose()

# # Set up a grid for subplots (e.g., 4 rows x 5 columns for 20 parameters)
# fig, axes = plt.subplots(4, 5, figsize=(20, 12), sharex=True, sharey=True)
# fig.suptitle("Individual MFCC Parameters", fontsize=16)

# # Plot each MFCC parameter in its own subplot
# for i in range(20):
#     row = i // 5  # Determine the row index in the 4x5 grid
#     col = i % 5   # Determine the column index in the 4x5 grid
#     sns.lineplot(data=mfcc_df[i], ax=axes[row, col])
#     axes[row, col].set_title(f'Parameter {i+1}')

# # Adjust layout to prevent overlapping labels
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for the main title
# plt.savefig('mfcc_individual_parameters.png')
# plt.show()


import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Load the MFCC data from a CSV file
mfcc_df = pd.read_csv('09-MFCC.csv', header=None)

# Transpose the data if needed
mfcc_df = mfcc_df.transpose()

# Create a 4x5 grid for the subplots
fig = make_subplots(rows=4, cols=5, subplot_titles=[f'Parameter {i+1}' for i in range(20)])

# Plot each parameter in its own subplot
for i in range(20):
    row = (i // 5) + 1  # Determine row in a 4x5 grid
    col = (i % 5) + 1   # Determine column in a 4x5 grid
    fig.add_trace(go.Scatter(y=mfcc_df[i][5000:5100], mode='lines', name=f'Parameter {i+1}'), row=row, col=col)

# Update layout to add titles and adjust spacing
fig.update_layout(
    height=800, width=1200,
    title_text="Individual MFCC Parameters (Interactive)",
    showlegend=False
)
fig.update_xaxes(title_text="Sample Index")
fig.update_yaxes(title_text="MFCC Value")

# Show the interactive plot
fig.show()

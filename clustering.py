import numpy as np
import pandas as pd
import librosa
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler

# Feature extraction as discussed earlier
features_list = []

for file in range(1, 116):
    if file < 10:
        filename = "0" + str(file) + "-MFCC.csv"
    else:
        filename = str(file) + "-MFCC.csv"
    
    # Load MFCCs from CSV
    mfcc_df = pd.read_csv(filename, header=None)

    mfcc_df = mfcc_df.transpose()
    
    # Extract the first three MFCCs
    mfcc_first_three = mfcc_df.values[:, :3]
    
    # Calculate delta and delta-delta of the first MFCC
    delta_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0])  # Delta of the 1st MFCC
    delta2_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0], order=2)  # Delta-delta of the 1st MFCC
    
    # Compute the mean and standard deviation for the 1st, 2nd, 3rd MFCC and the delta/delta-delta of the 1st MFCC
    feature_means = [
        np.mean(mfcc_first_three[:, 0]),  # Mean of 1st MFCC
        np.mean(mfcc_first_three[:, 1]),  # Mean of 2nd MFCC
        np.mean(mfcc_first_three[:, 2]),  # Mean of 3rd MFCC
        np.mean(delta_mfcc_1),            # Mean of delta of 1st MFCC
        np.mean(delta2_mfcc_1)            # Mean of delta-delta of 1st MFCC
    ]
    
    feature_stds = [
        np.std(mfcc_first_three[:, 0]),   # Std dev of 1st MFCC
        np.std(mfcc_first_three[:, 1]),   # Std dev of 2nd MFCC
        np.std(mfcc_first_three[:, 2]),   # Std dev of 3rd MFCC
        np.std(delta_mfcc_1),             # Std dev of delta of 1st MFCC
        np.std(delta2_mfcc_1)             # Std dev of delta-delta of 1st MFCC
    ]
    
    # Combine the mean and std features into one list (total 10 features per file)
    file_features = feature_means + feature_stds
    
    # Append the features for this file to the list
    features_list.append(file_features)

# Convert the list of features to a DataFrame
features_df = pd.DataFrame(features_list, columns=[
    'mean_mfcc1', 'mean_mfcc2', 'mean_mfcc3', 'mean_delta_mfcc1', 'mean_delta2_mfcc1',
    'std_mfcc1', 'std_mfcc2', 'std_mfcc3', 'std_delta_mfcc1', 'std_delta2_mfcc1'
])

# Save the features DataFrame to a CSV file
features_df.to_csv('features_dataframe.csv', index=False)

# Normalize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

# Apply different clustering algorithms
cluster_results = {}

# 1. KMeans Clustering
kmeans = KMeans(n_clusters=6, random_state=42)  # Choose an arbitrary number of clusters, e.g., 6
kmeans_labels = kmeans.fit_predict(scaled_features)
cluster_results['KMeans'] = kmeans_labels

# 2. DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)  # Adjust epsilon and min_samples as necessary
dbscan_labels = dbscan.fit_predict(scaled_features)
cluster_results['DBSCAN'] = dbscan_labels

# 3. Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=6)  # Again, choosing 6 clusters
agglo_labels = agglo.fit_predict(scaled_features)
cluster_results['Agglomerative'] = agglo_labels

# Save clustering results for later 

import numpy as np
import pandas as pd
import librosa
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin_min, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# --- Step 1: Feature Extraction ---
features_list = []

# Extract features from each file
for file in range(1, 116):
    if file < 10:
        filename = f"0{file}-MFCC.csv"
    else:
        filename = f"{file}-MFCC.csv"
    
    # Load MFCCs from CSV
    mfcc_df = pd.read_csv(filename, header=None).transpose()
    
    # Extract the first three MFCCs
    mfcc_first_three = mfcc_df.values[:, :3]
    
    # Calculate delta and delta-delta of the first MFCC
    delta_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0])
    delta2_mfcc_1 = librosa.feature.delta(mfcc_first_three[:, 0], order=2)
    
    # Calculate mean and std for the MFCCs and their deltas
    feature_means = [
        np.mean(mfcc_first_three[:, 0]), np.mean(mfcc_first_three[:, 1]), np.mean(mfcc_first_three[:, 2]),
        np.mean(delta_mfcc_1), np.mean(delta2_mfcc_1)
    ]
    feature_stds = [
        np.std(mfcc_first_three[:, 0]), np.std(mfcc_first_three[:, 1]), np.std(mfcc_first_three[:, 2]),
        np.std(delta_mfcc_1), np.std(delta2_mfcc_1)
    ]
    
    # Combine means and stds into one feature vector
    file_features = feature_means + feature_stds
    features_list.append(file_features)

# Convert to DataFrame
features_df = pd.DataFrame(features_list, columns=[
    'mean_mfcc1', 'mean_mfcc2', 'mean_mfcc3', 'mean_delta_mfcc1', 'mean_delta2_mfcc1',
    'std_mfcc1', 'std_mfcc2', 'std_mfcc3', 'std_delta_mfcc1', 'std_delta2_mfcc1'
])

# --- Step 2: Feature Scaling ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features_df)

n_clusters = 6
cluster_results = {}

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans_labels = kmeans.fit_predict(scaled_features)
kmeans_distances = kmeans.transform(scaled_features)

# Calculate likelihoods using inverse distances for KMeans
kmeans_likelihoods = 1 / (1 + kmeans_distances)
kmeans_likelihoods = kmeans_likelihoods / kmeans_likelihoods.sum(axis=1, keepdims=True)

# --- Gaussian Mixture Model (GMM) ---
gmm = GaussianMixture(n_components=n_clusters, random_state=42)
gmm.fit(scaled_features)
gmm_labels = gmm.predict(scaled_features)
gmm_probabilities = gmm.predict_proba(scaled_features)

# --- Agglomerative Clustering ---
agglo = AgglomerativeClustering(n_clusters=n_clusters)
agglo_labels = agglo.fit_predict(scaled_features)

# Calculate pairwise distances for Agglomerative Clustering
agglo_distances = np.zeros((len(scaled_features), n_clusters))
for i in range(n_clusters):
    cluster_points = scaled_features[agglo_labels == i]
    if len(cluster_points) > 0:
        _, distances = pairwise_distances_argmin_min(scaled_features, cluster_points)
        agglo_distances[:, i] = distances
    else:
        agglo_distances[:, i] = np.inf  # Handle empty clusters

agglo_likelihoods = 1 / (1 + agglo_distances)
agglo_likelihoods = agglo_likelihoods / agglo_likelihoods.sum(axis=1, keepdims=True)

# --- Step 3: Save Results ---
likelihoods_df = pd.DataFrame({
    'File_Number': list(range(1, 116)),
    'KMeans_Cluster': kmeans_labels,
    'GMM_Cluster': gmm_labels,
    'Agglomerative_Cluster': agglo_labels
})

# Append KMeans likelihoods for each cluster
for i in range(n_clusters):
    likelihoods_df[f'KMeans_Likelihood_Cluster_{i+1}'] = kmeans_likelihoods[:, i]

# Append GMM likelihoods for each cluster
for i in range(n_clusters):
    likelihoods_df[f'GMM_Likelihood_Cluster_{i+1}'] = gmm_probabilities[:, i]

# Append Agglomerative Clustering likelihoods for each cluster
for i in range(n_clusters):
    likelihoods_df[f'Agglomerative_Likelihood_Cluster_{i+1}'] = agglo_likelihoods[:, i]

# Save to CSV
likelihoods_df.to_csv('clustering_full_likelihoods.csv', index=False)
print("Full likelihood results saved to 'clustering_full_likelihoods.csv'.")

# --- Step 4: Calculate Clustering Metrics ---
# Silhouette Score (for overall clustering quality)
kmeans_silhouette = silhouette_score(scaled_features, kmeans_labels)
gmm_silhouette = silhouette_score(scaled_features, gmm_labels)
agglo_silhouette = silhouette_score(scaled_features, agglo_labels)

# Davies-Bouldin Index (lower values are better)
kmeans_davies_bouldin = davies_bouldin_score(scaled_features, kmeans_labels)
gmm_davies_bouldin = davies_bouldin_score(scaled_features, gmm_labels)
agglo_davies_bouldin = davies_bouldin_score(scaled_features, agglo_labels)

# Calinski-Harabasz Index (higher values are better)
kmeans_calinski = calinski_harabasz_score(scaled_features, kmeans_labels)
gmm_calinski = calinski_harabasz_score(scaled_features, gmm_labels)
agglo_calinski = calinski_harabasz_score(scaled_features, agglo_labels)

# Display the results
print("\nClustering Evaluation Metrics:")

# Silhouette Scores
print(f"KMeans Silhouette Score: {kmeans_silhouette:.4f}")
print(f"GMM Silhouette Score: {gmm_silhouette:.4f}")
print(f"Agglomerative Clustering Silhouette Score: {agglo_silhouette:.4f}")

# Davies-Bouldin Index
print(f"KMeans Davies-Bouldin Index: {kmeans_davies_bouldin:.4f}")
print(f"GMM Davies-Bouldin Index: {gmm_davies_bouldin:.4f}")
print(f"Agglomerative Clustering Davies-Bouldin Index: {agglo_davies_bouldin:.4f}")

# Calinski-Harabasz Index
print(f"KMeans Calinski-Harabasz Index: {kmeans_calinski:.4f}")
print(f"GMM Calinski-Harabasz Index: {gmm_calinski:.4f}")
print(f"Agglomerative Clustering Calinski-Harabasz Index: {agglo_calinski:.4f}")

# --- Visualization using t-SNE ---
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
X_tsne = tsne.fit_transform(scaled_features)
tsne_df = pd.DataFrame(X_tsne, columns=['t-SNE1', 't-SNE2'])
tsne_df['KMeans_Cluster'] = kmeans_labels
tsne_df['GMM_Cluster'] = gmm_labels
tsne_df['Agglomerative_Cluster'] = agglo_labels

def plot_tsne(df, cluster_column, title):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='t-SNE1', y='t-SNE2', hue=cluster_column, data=df, palette='viridis', edgecolor='k')
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

# Plot t-SNE results
plot_tsne(tsne_df, 'KMeans_Cluster', 't-SNE: KMeans Clustering')
plot_tsne(tsne_df, 'GMM_Cluster', 't-SNE: GMM Clustering')
plot_tsne(tsne_df, 'Agglomerative_Cluster', 't-SNE: Agglomerative Clustering')

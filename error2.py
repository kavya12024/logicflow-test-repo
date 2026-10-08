import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Generate and Save a Dummy Dataset to CSV
X, _ = make_blobs(n_samples=400, n_features=2, centers=4, cluster_std=1.2, random_state=42)
df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
df.to_csv('Heart.csv', index=False)

# 2. Load and Preprocess Data
data = pd.read_csv('Heart.csv')  # Corrected 'ad_csv' to 'read_csv'
X_scaled = StandardScaler().fit_transform(data)

# 3. The Elbow Method to Find Optimal K
wcss = []
k_range = range(1, 11)

for k in k_range:
    kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans_temp.fit(X_scaled)
    wcss.append(kmeans_temp.inertia_)

# Plotting the Elbow Curve to find optimal K visually
plt.figure(figsize=(7, 4))
plt.plot(k_range, wcss, marker='o', linestyle='--', color='b')
plt.title('Elbow Method For Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS (Inertia)')
plt.grid(True)
plt.show()

# Based on the dataset generation, the sharp "elbow" point is at K = 4
optimal_k = 4
print(f"Chosen Optimal K from Elbow Method: {optimal_k}\n")

# 4. Final K-Means Execution with Optimal K
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)

# Find Means / Centroids
centroids = kmeans.cluster_centers_
print("--- K-Means Cluster Centroids (Scaled Coordinates) ---")
for i, center in enumerate(centroids):
    print(f"Cluster {i} Centroid: {center}")
print("")

# Calculate distance of each data point to its assigned centroid
# kmeans.transform returns Euclidean distance to all centroids
distances_to_all_centroids = kmeans.transform(X_scaled)
distances_to_assigned_centroid = np.min(distances_to_all_centroids, axis=1)

# Add results to a summary dataframe
data['Cluster_Label'] = kmeans_labels
data['Distance_to_Centroid'] = distances_to_assigned_centroid
print("--- Sample of Data Points with Centroid Distances ---")
print(data.head(5), "\n")

# 5. EM (Gaussian Mixture Model) Execution with Optimal K
em = GaussianMixture(n_components=optimal_k, random_state=42)
em_labels = em.fit_predict(X_scaled)

# 6. Quality Metrics Evaluation
kmeans_score = silhouette_score(X_scaled, kmeans_labels)
em_score = silhouette_score(X_scaled, em_labels)

print("--- Clustering Quality Performance ---")
print(f"K-Means Silhouette Score: {kmeans_score:.4f}")
print(f"EM (GMM) Silhouette Score : {em_score:.4f}\n")

# 7. Draw the Clusters for Comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K-Means Visualisation
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis', s=40, alpha=0.6)
axes[0].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=250, label='Centroids')
axes[0].set_title(f'K-Means Drawing (K={optimal_k})\nSilhouette Score: {kmeans_score:.3f}')
axes[0].legend()
axes[0].grid(True)

# EM Visualisation
axes[1].scatter(X_scaled[:, 0], X_scaled[:, 1], c=em_labels, cmap='plasma', s=40, alpha=0.6)
axes[1].set_title(f'EM / GMM Drawing (K={optimal_k})\nSilhouette Score: {em_score:.3f}')
axes[1].grid(True)

plt.tight_layout()
plt.show()
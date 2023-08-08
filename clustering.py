import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random data points in three dimensions
np.random.seed(0)
num_samples = 400
num_clusters = 4

# Generate data for each cluster
cluster_centers = np.array([
    [5e-6, 2e-1, 10e7],
    [3e-6, 4e-1, 15e7],
    [7e-6, 3e-1, 5e7],
    [6e-6, 4e-1, 12e7]
])

data = np.zeros((num_samples, 3))

for i in range(num_clusters):
    start_idx = i * (num_samples // num_clusters)
    end_idx = (i + 1) * (num_samples // num_clusters)
    data[start_idx:end_idx, 0] = cluster_centers[i, 0] + np.random.normal(0, 0.75e-6, (end_idx - start_idx))
    data[start_idx:end_idx, 1] = cluster_centers[i, 1] + np.random.normal(0, 0.6e-1, (end_idx - start_idx))
    data[start_idx:end_idx, 2] = cluster_centers[i, 2] + np.random.normal(0, 0.8e7, (end_idx - start_idx))

# Perform K-means clustering
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', edgecolor='k')

# Plot cluster centroids
ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], c='red', marker='x', s=200)

ax.set_xlabel('Pulse Width ')

xtick_positions = [2e-6, 4e-6, 6e-6, 8e-6, 10e-6]
xtick_labels = ['{:.2e}'.format(pos) for pos in xtick_positions]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(xtick_labels)
ax.set_ylabel('Pulse Amplitude')
ax.set_zlabel('Frequency')
ax.set_title('RF Pulse clustering using 3 features')
plt.savefig("./plots/cluster.png")
plt.show()

import numpy as np
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random data points in three dimensions
np.random.seed(0)
num_samples = 400

# Generate data for each cluster
cluster_centers = np.array([
    [5e-6, 2e-1, 10e7],
    [3e-6, 4e-1, 15e7],
    [7e-6, 3e-1, 5e7],
    [6e-6, 4e-1, 12e7]
])

data = np.zeros((num_samples, 3))

for i in range(len(cluster_centers)):
    start_idx = i * (num_samples // len(cluster_centers))
    end_idx = (i + 1) * (num_samples // len(cluster_centers))
    data[start_idx:end_idx, 0] = cluster_centers[i, 0] + np.random.normal(0, 0.75e-6, (end_idx - start_idx))
    data[start_idx:end_idx, 1] = cluster_centers[i, 1] + np.random.normal(0, 0.6e-1, (end_idx - start_idx))
    data[start_idx:end_idx, 2] = cluster_centers[i, 2] + np.random.normal(0, 0.8e7, (end_idx - start_idx))

# Perform DBSCAN clustering
epsilon = 2e7  # Maximum distance between samples to be considered neighbors
min_samples = 10  # Minimum number of samples in a neighborhood to form a core point
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
labels = dbscan.fit_predict(data)
num_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # Number of clusters, ignoring noise points
print(f"Num clusters = {num_clusters}")
# Plotting
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=labels, cmap='viridis', edgecolor='k')

ax.set_xlabel('Pulse Width')
ax.set_ylabel('Pulse Amplitude')
ax.set_zlabel('Frequency')
ax.set_title('RF Pulse clustering')
plt.savefig("./plots/cluster_dbscan.png")
plt.show()

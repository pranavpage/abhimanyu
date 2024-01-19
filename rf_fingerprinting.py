import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

pdw_table = pd.read_csv("pdw_table.csv")
# Select the features for clustering
features = pdw_table[["pulse_width"]]

# Perform K-means clustering
num_clusters = 3  # Number of clusters
cluster_labels = KMeans(n_clusters=num_clusters, random_state=0).fit_predict(features)

# Add the cluster labels to the dataframe
pdw_table["cluster"] = cluster_labels
pdw_table.to_csv("pdw_table.csv", index=False)
# Plot the data points colored by clusters
# plt.figure(figsize=(10, 6))
# plt.scatter(pdw_table["pulse_width"]*1e6, pdw_table["pulse_max_power"], c=pdw_table["cluster"], cmap="viridis")
# plt.xlabel("Pulse Width (us)")
# plt.ylabel("Pulse Max Power")
# plt.grid(True)
# plt.title("Classification of Radar Pulses")
# plt.colorbar(label="Cluster")
# plt.savefig("./plots/rfp_cluster.png")
df0 = pdw_table[pdw_table['cluster']==0]
df1 = pdw_table[pdw_table['cluster']==1]
df2 = pdw_table[pdw_table['cluster']==2]

plt.scatter(df0.diff().t_arrival[1:]*1e3, df0.cluster[1:])
plt.scatter(df1.diff().t_arrival[1:]*1e3, df1.cluster[1:])
plt.scatter(df2.diff().t_arrival[1:]*1e3, df2.cluster[1:])
plt.yticks([0, 1, 2])
plt.grid()
plt.xlabel("Pulse Repetition Interval (ms)")
plt.ylabel("Cluster Number")
plt.title("PRI Extraction after deinterleaving")
plt.savefig("./plots/rfp_pri.png")
# X = pdw_table[["pulse_width", "pulse_max_power"]]
# y = pdw_table["cluster"]

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# # Initialize the Support Vector Classifier
# clf = SVC(kernel='linear', C=10, random_state=0)

# # Train the classifier on the training data
# clf.fit(X_train, y_train)

# # Predict the clusters on the testing data
# y_pred = clf.predict(X_test)

# # Calculate the accuracy of the classifier
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
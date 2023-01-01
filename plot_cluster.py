import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
est = pd.read_csv("combined_data.csv", header=None, names=["theta", "phi"])
true_pos = [1.047197, 0.698132]
plt.figure(0)
plt.scatter(est["theta"], est["phi"], alpha=0.7)
plt.scatter(true_pos[0], true_pos[1], alpha=1, c='red', label="True location")
plt.grid()
plt.xlabel("Theta (rad)")
plt.ylabel("Phi (rad)")
plt.title("Estimates of location using BM+FM method")
plt.legend()
plt.savefig("plot_cluster.png")
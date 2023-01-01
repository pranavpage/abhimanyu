import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
est = pd.read_csv("combined_data.csv", header=None, names=["theta", "phi"])
true_pos = [1.047197, 0.698132]
R_e = 6.3781e6
plt.figure(0)
plt.scatter(est["theta"], est["phi"], alpha=0.5)
plt.scatter(true_pos[0], true_pos[1], alpha=1, c='red', label="True location")
plt.scatter(np.mean(est["theta"]), np.mean(est["phi"]), c='green', marker='x', label="Mean estimate")
plt.grid()
# plt.xlim(true_pos[0]-0.005, true_pos[0]+0.005)
# plt.ylim(true_pos[1]-0.005, true_pos[1]+0.005)
# plt.xticks(np.linspace(true_pos[0]-0.005, true_pos[0]+0.005, 5))
plt.xlabel("Theta (rad)")
plt.ylabel("Phi (rad)")
plt.title("Estimates of location using BM+FM method")
plt.legend()
plt.savefig("plot_cluster_combined.png")
stddev = np.std(est)
print(f"Stddev for theta = {stddev['theta']}, Stddev for phi = {stddev['phi']}")
print(f"90% confidence intervals : X = {R_e*np.sin(true_pos[0])*1.645*stddev['phi']/1e3:.6f} km, Y = {R_e*1.645*stddev['theta']/1e3:.6f} km")
# plt.figure(1)
# plt.hist2d(est["theta"], est["phi"], cmap = plt.cm.nipy_spectral)
# plt.title("2D Histogram of estimates using BM+FM method")
# plt.colorbar()
# plt.xlabel("Theta (rad)")
# plt.ylabel("Phi (rad)")
# plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib.patches import Ellipse
from config import *
def theta2lat(theta):
    return (90 - np.degrees(theta))
est = pd.read_csv(f"./data/{tag}.csv", header=None, names=["theta", "phi"])
def ellipse_dim(x,y):
    x = np.degrees(x)
    y = theta2lat(y)
    covariance_matrix = np.cov(x, y)
    eig_val, eig_vec = np.linalg.eig(covariance_matrix)
    lambda1 = eig_val[0]
    lambda2 = eig_val[1]
    a_ellipse = 2*2*np.sqrt(lambda1)
    b_ellipse = 2*2*np.sqrt(lambda2)
    angle = np.degrees(np.arctan2(*eig_vec[:,0][::-1]))
    return a_ellipse, b_ellipse, angle
true_pos = [1.047197, 0.698132]
R_e = 6.3781e6
plt.scatter(np.degrees(est["phi"]), theta2lat(est["theta"]), alpha=0.5, label="Estimate")
plt.scatter(np.degrees(true_pos[1]), theta2lat(true_pos[0]), alpha=1, c='red', label="True location")
plt.scatter(np.degrees(np.mean(est["phi"])), theta2lat(np.mean(est["theta"])), c='green', marker='x', label="Mean estimate")
plt.grid()
stddev = np.std(est)
spread = 0.01
plt.xlim(np.degrees(true_pos[1]-spread), np.degrees(true_pos[1]+spread))
plt.ylim(theta2lat(true_pos[0]-spread), theta2lat(true_pos[0]+spread))
# plt.xticks(np.linspace(true_pos[0]-0.005, true_pos[0]+0.005, 5))
plt.xlabel("Longitude (degrees)")
plt.ylabel("Latitude (degrees)")
# plt.title(f"Estimates of location using BM+FM method\n95% confidence intervals : X = {R_e*np.sin(true_pos[0])*2*stddev['phi']/1e3:.2f} km, Y = {R_e*2*stddev['theta']/1e3:.2f} km\nlongitude offset = 0 degrees")
plt.subplots_adjust(top=0.85, left=0.15)
# print(f"Stddev for theta = {stddev['theta']}, Stddev for phi = {stddev['phi']}")
# print(f"95% confidence intervals : X = {R_e*np.sin(true_pos[0])*2*stddev['phi']/1e3:.6f} km, Y = {R_e*2*stddev['theta']/1e3:.6f} km")
x = np.degrees(est["phi"]) # theta
y = theta2lat(est["theta"]) # phi
covariance_matrix = np.cov(x, y)
eig_val, eig_vec = np.linalg.eig(covariance_matrix)
lambda1 = eig_val[0]
lambda2 = eig_val[1]
a_ellipse = 2*2*np.sqrt(lambda1)
b_ellipse = 2*2*np.sqrt(lambda2)
angle = np.degrees(np.arctan2(*eig_vec[:,0][::-1]))
ellipse2 = Ellipse(xy=(np.mean(x),np.mean(y)),
                   width=2*2*np.sqrt(lambda1),
                   height=2*2*np.sqrt(lambda2),
                   angle=angle,
                   fill=False,
                   alpha=1,
                   ec='black', lw=1, label="95$\%$ confidence region")
ax = plt.gca()
ax.add_patch(ellipse2)
plt.title(f"Estimates of location using {method} method, {num_sats} sats\n95% confidence region : width = {R_e*np.radians(a_ellipse)/1e3:.2f} km, height = {R_e*np.radians(b_ellipse)/1e3:.2f} km, area = {np.pi*np.radians(a_ellipse)/2*np.radians(b_ellipse)/2*(R_e**2)/1e6:.2f} sq km", fontsize=10)
plt.legend()
plt.savefig(f"./plots/{tag}.png")
# plt.xlabel('x', fontsize=15)
# plt.ylabel('y', fontsize=15)
# plt.show()
# plt.figure(1)
# bm_est = pd.read_csv("bm_data.csv", header=None, names=["theta", "phi"])
# fm_est = pd.read_csv("fm_data_nm.csv" , header=None, names=["theta", "phi"])
# combined_est = pd.read_csv("combined_data.csv", header=None, names=["theta", "phi"])
# plt.scatter(np.degrees(bm_est["phi"]), theta2lat(bm_est["theta"]), alpha=0.3, label="BM Estimate", c='blue')
# plt.scatter(np.degrees(fm_est["phi"]), theta2lat(fm_est["theta"]), alpha=0.3, label="FM Estimate", c='limegreen')
# plt.scatter(np.degrees(true_pos[1]), theta2lat(true_pos[0]), alpha=1, c='red', label="True location")
# plt.scatter(np.degrees(est["phi"]), theta2lat(est["theta"]), alpha=0.9, label="BM+FM Estimate", c='orange')

# bm_a, bm_b, bm_angle = ellipse_dim(bm_est['phi'], bm_est['theta'])
# fm_a, fm_b, fm_angle = ellipse_dim(fm_est['phi'], fm_est['theta'])
# combined_a, combined_b, combined_angle = ellipse_dim(combined_est['phi'], combined_est['theta'])
# bm_ellipse = Ellipse(xy=(np.mean(np.degrees(bm_est['phi'])),np.mean(theta2lat(bm_est['theta']))),
#                    width=bm_a,
#                    height=bm_b,
#                    angle=bm_angle,
#                    fill=False,
#                    alpha=1,
#                    ec='blue', lw=2)
# fm_ellipse = Ellipse(xy=(np.mean(np.degrees(fm_est['phi'])),np.mean(theta2lat(fm_est['theta']))),
#                    width=fm_a,
#                    height=fm_b,
#                    angle=fm_angle,
#                    fill=False,
#                    alpha=1,
#                    ec='limegreen', lw=2)
# combined_ellipse = Ellipse(xy=(np.mean(np.degrees(combined_est['phi'])),np.mean(theta2lat(combined_est['theta']))),
#                    width=combined_a,
#                    height=combined_b,
#                    angle=combined_angle,
#                    fill=False,
#                    alpha=1,
#                    ec='orange', lw=2)
# ax = plt.gca()
# ax.add_patch(bm_ellipse)
# ax.add_patch(fm_ellipse)
# ax.add_patch(combined_ellipse)
# spread = 0.003
# plt.title("Comparison of performance of BM, FM and Combined methods")
# plt.xlim(np.degrees(true_pos[1]-spread), np.degrees(true_pos[1]+spread))
# plt.ylim(theta2lat(true_pos[0]-spread), theta2lat(true_pos[0]+spread))
# plt.legend()
# plt.xlabel("Longitude (degrees)")
# plt.ylabel("Latitude (degrees)")
# # plt.title(f"Estimates of location using BM+FM method\n95% confidence intervals : X = {R_e*np.sin(true_pos[0])*2*stddev['phi']/1e3:.2f} km, Y = {R_e*2*stddev['theta']/1e3:.2f} km\nlongitude offset = 0 degrees")
# plt.subplots_adjust(top=0.85, left=0.15)
# plt.grid()
# plt.savefig("plot_cluster_comparison.png")
# plt.show()
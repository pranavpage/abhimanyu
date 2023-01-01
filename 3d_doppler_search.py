import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import scipy.optimize as so
#########################################
# Data generation
def cart(s_pos):
    return np.array([s_pos[0]*np.sin(s_pos[1])*np.cos(s_pos[2]), s_pos[0]*np.sin(s_pos[1])*np.sin(s_pos[2]), s_pos[0]*np.cos(s_pos[1])])
R_e = 6.3781e6
lat_e = 30
long_e = 40
f_c = 200e6
h = 500e3
R_s = R_e + h
G = sc.gravitational_constant
M = 5.9722e24
v_s = np.sqrt(G*M/(R_e+h))
max_slant_angle = np.arcsin(R_e/(R_e+h))
max_slant_range = R_s*np.cos(max_slant_angle)
orbital_period = 2*np.pi*R_s/(v_s)
angular_velocity = 2*np.pi/(orbital_period)
polar_e = [R_e, np.radians(90-lat_e), np.radians(long_e)]
print(f"Location of source : {lat_e:.2f} lat, {long_e:.2f} long")
print(f"Location of source (radians) : theta = {polar_e[1]} , phi = {polar_e[2]}")
num_sample_points = 100
num_grid_points = 20
measurement_period = 60 # in seconds
lat_start_s = lat_e -10
long_s = long_e + 5
polar_start_s = [R_s, np.radians(90-lat_start_s), np.radians(long_s)]
print(f"Altitude of satellite : {h/1e3:.1f} km \nVelocity = {v_s*3600/1e3:.1f} kmph")
print(f"Orbital period : {orbital_period/3600:.2f} hours")
print(f"Angular velocity : {angular_velocity*180/np.pi:.3f} deg/s")
print(f"Max slant angle : {np.degrees(max_slant_angle):.2f} degrees")
print(f"Center Frequency = {f_c/1e6:.2f} MHz")
print(f"Max slant range : {max_slant_range/1e3:.2f} km")
print(f"Footprint area : {2*np.pi*(R_e**2)*(1-np.cos(np.pi/2 - max_slant_angle))/1e6:.2f} sq km")
cart_e = cart(polar_e)
def orbital_params(t):
    # returns [theta_s, phi_s, vx, vy, vz] of satellite
    theta_s_i = polar_start_s[1] - angular_velocity*t
    phi_s_i = polar_start_s[2]
    cart_s_i = cart([polar_start_s[0], theta_s_i, phi_s_i])
    polar_vpoint = [polar_start_s[0]/np.cos(theta_s_i),0,0]
    cart_vpoint = cart(polar_vpoint)
    [vx_s, vy_s, vz_s] = (cart_vpoint - cart_s_i)*v_s/np.linalg.norm((cart_vpoint - cart_s_i), 2)
    return [theta_s_i, phi_s_i, vx_s, vy_s, vz_s]
def doppler_shift(o_params, cart_e = cart_e):
    [theta_s, phi_s, vx_s, vy_s, vz_s] = o_params
    [x_s, y_s, z_s] = list(cart([R_s, theta_s, phi_s]))
    cart_es = np.array([x_s, y_s, z_s]) - cart_e
    r_es = np.linalg.norm(cart_es)
    fraction_shift = (1 - (vx_s*(x_s - cart_e[0]) + vy_s*(y_s - cart_e[1]) + vz_s*(z_s - cart_e[2]))/(sc.c*r_es))
    return fraction_shift
measurement_times = np.linspace(0, measurement_period, num_sample_points)
psi = np.array([doppler_shift(orbital_params(t)) for t in measurement_times])
###################################################################
# Estimation

def doppler_cost(pos, psi = psi):
    # theta_k, phi_k is the candidate location in the grid search
    [theta_k, phi_k] = pos
    cart_k = cart([R_e, theta_k, phi_k])
    psi_k = np.array([doppler_shift(orbital_params(t), cart_e=cart_k) for t in measurement_times])
    return (np.linalg.norm(psi_k-psi, 2))**2
print(f"Doppler cost at original location = {doppler_cost([polar_e[1], polar_e[2]])}")
# grid search 
def grid_search(psi=psi):
    cost_arr = []
    for i in range(num_grid_points_theta):
        for j in range(num_grid_points_phi):
            theta_i = theta_arr[i]
            phi_j = phi_arr[j]
            cost_arr.append([doppler_cost([theta_i, phi_j], psi), theta_i, phi_j])
    return cost_arr
num_grid_points_theta = 20
num_grid_points_phi = 2*num_grid_points_theta
footprint_width = np.pi/2 - max_slant_angle
print(f"Footprint width = {footprint_width:.4f}")
theta_min = polar_start_s[1] - footprint_width
theta_max = polar_start_s[1] + footprint_width
print(f"Theta min = {theta_min:.4f}, Theta max = {theta_max:.4f}")
phi_min = polar_start_s[2] - footprint_width/np.cos(polar_start_s[1])
phi_max = polar_start_s[2] + footprint_width/np.cos(polar_start_s[1])
theta_arr = np.linspace(theta_min, theta_max, num_grid_points_theta)
phi_arr = np.linspace(phi_min, phi_max, num_grid_points_phi)
num_top = 5
cost_arr = np.array(grid_search(psi))
cost_argsort = np.argsort(cost_arr[:,0])
for i in range(num_top):
    i_entry = cost_arr[cost_argsort[i]]
    print(f"{i}: Cost={i_entry[0]:.3e}, theta={i_entry[1]:.4f}, phi={i_entry[2]:.4f}")
    res = so.minimize(doppler_cost, i_entry[1:], method='Nelder-Mead')
    print(res.x, res.fun)
print(f"Location of source (radians) : theta = {polar_e[1]} , phi = {polar_e[2]}")
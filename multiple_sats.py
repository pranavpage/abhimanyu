import numpy as np
import scipy.constants as sc
import matplotlib.pyplot as plt
import scipy.optimize as so
import pandas as pd
from config import *
######### Orbital Parameters ###########
R_e = 6.3781e6
f_c = 200e6
h = 500e3
R_s = R_e + h
G = sc.gravitational_constant
M = 5.9722e24
c = 2.99792458e8
v_s = np.sqrt(G*M/(R_e+h))
max_slant_angle = np.arcsin(R_e/(R_e+h))
max_slant_range = R_s*np.cos(max_slant_angle)
orbital_period = 2*np.pi*R_s/(v_s)
angular_velocity = 2*np.pi/(orbital_period)
polar_e = [R_e, np.radians(90-lat_e), np.radians(long_e)]
print(f"True position of emitter = ({polar_e[1], polar_e[2]})")
measurement_period = 60 # in seconds
lat_start_s = lat_e - 10
long_s = long_e + 10
polar_start_s = [R_s, np.radians(90-lat_start_s), np.radians(long_s)]
lat_s_arr = np.linspace(lat_start_s+10, lat_start_s-10, num_sats)
long_s_arr = np.linspace(long_s, long_s-20, num_sats)
polar_sats_arr = [[R_s, np.radians(90-lat_s_arr[i]), np.radians(long_s_arr[i])] for i in range(num_sats)]
# print([f"({lat_s_arr[i]:.2f}, {long_s_arr[i]:.2f})" for i in range(num_sats)])
# print(polar_sats_arr)
def cart(s_pos):
    return np.array([s_pos[0]*np.sin(s_pos[1])*np.cos(s_pos[2]), s_pos[0]*np.sin(s_pos[1])*np.sin(s_pos[2]), s_pos[0]*np.cos(s_pos[1])])
cart_e = cart(polar_e)
measurement_period = 60 # in seconds

######## Search Parameters ########
cart_e = cart(polar_e)
num_grid_points_theta = 10
num_grid_points_phi = 2*num_grid_points_theta
footprint_width = np.pi/2 - max_slant_angle
# print(f"Footprint width = {footprint_width:.4f} rad = {np.degrees(footprint_width):.4f} deg")
theta_min = polar_start_s[1] - footprint_width
theta_max = polar_start_s[1] + footprint_width
# print(f"Theta min = {theta_min:.4f}, Theta max = {theta_max:.4f}")
phi_min = polar_start_s[2] - footprint_width/np.cos(polar_start_s[1])
phi_max = polar_start_s[2] + footprint_width/np.cos(polar_start_s[1])
theta_arr = np.linspace(theta_min, theta_max, num_grid_points_theta)
phi_arr = np.linspace(phi_min, phi_max, num_grid_points_phi)
measurement_times = np.linspace(0, measurement_period, num_sample_points)
############### Noise Params ###############
std_bearing = np.radians(std_bearing_deg)

def orbital_params(polar_sat, t):
    # returns [theta_s, phi_s, vx, vy, vz] of satellite
    theta_s_i = polar_sat[1] - angular_velocity*t
    phi_s_i = polar_sat[2]
    cart_s_i = cart([polar_sat[0], theta_s_i, phi_s_i])
    polar_vpoint = [polar_sat[0]/np.cos(theta_s_i),0,0]
    cart_vpoint = cart(polar_vpoint)
    [vx_s, vy_s, vz_s] = (cart_vpoint - cart_s_i)*v_s/np.linalg.norm((cart_vpoint - cart_s_i), 2)
    return [theta_s_i, phi_s_i, vx_s, vy_s, vz_s]

def bearing_angles(o_params, cart_e):
    [theta_s, phi_s, vx_s, vy_s, vz_s] = o_params
    [x_s, y_s, z_s] = list(cart([R_s, theta_s, phi_s]))
    cart_s = np.array([x_s, y_s, z_s])
    cart_es = cart_e - cart_s
    cart_vs = np.array([vx_s, vy_s, vz_s])
    XSAT = cart_vs/np.linalg.norm(cart_vs, 2)
    ZSAT = cart_s/np.linalg.norm(cart_s, 2)
    YSAT = np.cross(ZSAT, XSAT)
    theta_b = np.arccos(np.dot(cart_es, ZSAT)/np.linalg.norm(cart_es, 2))
    vec_in_plane = cart_es - np.dot(cart_es, ZSAT)*cart_es/np.linalg.norm(cart_es, 2)    
    phi_b = np.arccos(np.dot(vec_in_plane, YSAT)/(np.linalg.norm(vec_in_plane, 2)))
    return [theta_b, phi_b]

def time_of_arrival(o_params, cart_e):
    [theta_s, phi_s, vx_s, vy_s, vz_s] = o_params
    [x_s, y_s, z_s] = list(cart([R_s, theta_s, phi_s]))
    cart_s = np.array([x_s, y_s, z_s])
    cart_es = cart_e - cart_s
    del_t = np.linalg.norm(cart_es, 2)/c
    return del_t

def doppler_shift(o_params, cart_e):
    [theta_s, phi_s, vx_s, vy_s, vz_s] = o_params
    [x_s, y_s, z_s] = list(cart([R_s, theta_s, phi_s]))
    cart_es = np.array([x_s, y_s, z_s]) - cart_e
    r_es = np.linalg.norm(cart_es)
    fraction_shift = (1 - (vx_s*(x_s - cart_e[0]) + vy_s*(y_s - cart_e[1]) + vz_s*(z_s - cart_e[2]))/(sc.c*r_es))
    return fraction_shift*f_c

def generate_psi_tdoa(polar_sats_arr=polar_sats_arr, cart_e=cart_e, measurement_times=measurement_times):
    psi_t = []
    for t in measurement_times:
        toa_arr = [time_of_arrival(orbital_params(polar_sats_arr[i], t), cart_e) for i in range(num_sats)]
        tdoa_arr = [(toa_arr[i+1] - toa_arr[i]) for i in range(num_sats-1)]
        psi_t.append(tdoa_arr)
    psi_t = np.array(psi_t)
    return psi_t
def generate_psi_fdoa(polar_sats_arr=polar_sats_arr, cart_e=cart_e, measurement_times=measurement_times):
    psi_f = []
    for t in measurement_times:
        foa_arr = [doppler_shift(orbital_params(polar_sats_arr[i], t), cart_e) for i in range(num_sats)]
        fdoa_arr = [(foa_arr[i+1] - foa_arr[i]) for i in range(num_sats-1)]
        psi_f.append(fdoa_arr)
    psi_f = np.array(psi_f)
    return psi_f
def generate_psi(method=method):
    if(method=="AoA"):
        psi_b = np.array([[bearing_angles(orbital_params(polar_sats_arr[i], t), cart_e) for i in range(num_sats)] for t in measurement_times])
        psi = psi_b
    elif(method=="TDoA"):
        # toa_arr = [[time_of_arrival(orbital_params(polar_sats_arr[i], t), cart_e) for i in range(num_sats)]]
        psi = generate_psi_tdoa()
    elif(method=="FDoA"):
        psi = generate_psi_fdoa()
    return psi

def bearing_cost(pos, psi_b):
    # pos : candidate location in grid search
    # psi_b : measurement vector for bearing angles
    # print("Bearing Cost")
    [theta_k, phi_k] = pos
    cart_k = cart([R_e, theta_k, phi_k])
    psi_b_k = np.array([[bearing_angles(orbital_params(polar_sats_arr[i], t), cart_k) for i in range(num_sats)] for t in measurement_times])
    psi_diff = psi_b - psi_b_k
    cost = (np.linalg.norm(psi_diff.flatten()))**2
    # print(cost, pos, polar_e[1:])
    return cost

def tdoa_cost(pos, psi_t):
    # pos : candidate location in grid search
    # psi_t : measurement vector for tdoa
    [theta_k, phi_k] = pos
    cart_k = cart([R_e, theta_k, phi_k])
    psi_t_k = generate_psi_tdoa(cart_e=cart_k)
    psi_diff = psi_t - psi_t_k
    cost = (np.linalg.norm(psi_diff.flatten()))**2
    # print(cost, pos, polar_e[1:])
    return cost


def fdoa_cost(pos, psi_t):
    # pos : candidate location in grid search
    # psi_f : measurement vector for fdoa
    [theta_k, phi_k] = pos
    cart_k = cart([R_e, theta_k, phi_k])
    psi_t_k = generate_psi_fdoa(cart_e=cart_k)
    psi_diff = psi_t - psi_t_k
    cost = (np.linalg.norm(psi_diff.flatten()))**2
    # print(cost, pos, polar_e[1:])
    return cost
def target_cost(pos, psi, method = method):
    if(method == "AoA"):
        # print("AoA")
        return bearing_cost(pos, psi_b=psi)
    elif(method == "TDoA"):
        return tdoa_cost(pos, psi_t=psi)
    elif(method == "FDoA"):
        return fdoa_cost(pos, psi_t=psi)
def combined_cost(pos, psi_1, psi_2, methods=["TDoA", "FDoA"]):
    cost_1 = target_cost(pos, psi_1, methods[0])
    cost_2 = target_cost(pos, psi_2, methods[1])
    w1 = weights[methods[0]]
    w2 = weights[methods[1]]
    return (w1*cost_1 + w2*cost_2)
def grid_search(num_grid_points_theta, num_grid_points_phi, psi_1, psi_2):
    cost_arr = []
    for i in range(num_grid_points_theta):
        for j in range(num_grid_points_phi):
            theta_i = theta_arr[i]
            phi_j = phi_arr[j]
            # cost_arr.append([target_cost([theta_i, phi_j], psi=psi), theta_i, phi_j])
            cost_arr.append([combined_cost([theta_i, phi_j], psi_1, psi_2), theta_i, phi_j])
    return np.array(cost_arr)
def std_method(method=method):
    if(method=="AoA"):
        std = std_bearing
    elif(method=="TDoA"):
        std = std_tdoa
    elif(method=="FDoA"):
        std = std_fdoa
    return std
print(f"Target (theta, phi) = {polar_e[1:]}")
results = []
psi_1 = generate_psi("TDoA")
psi_2 = generate_psi("FDoA")
for j in range(num_simulations):
    noise_1 = np.random.normal(loc=0, scale=std_method("TDoA"), size=psi_1.shape)
    noise_2 = np.random.normal(loc=0, scale=std_method("FDoA"), size=psi_2.shape)
    # psi_noisy = psi + noise
    num_top = 3
    cost_arr = np.array(grid_search(num_grid_points_theta, num_grid_points_phi, psi_1+noise_1, psi_2+noise_2))
    cost_argsort = np.argsort(cost_arr[:,0])
    top_arr = []
    for i in range(num_top):
        i_entry = cost_arr[cost_argsort[i]]
        print(f"{i}: Cost={i_entry[0]:.3e}, theta={i_entry[1]:.4f}, phi={i_entry[2]:.4f}")
        res = so.minimize(combined_cost, i_entry[1:], method='Nelder-Mead', args=(psi_1+noise_1, psi_2+noise_2))
        top_arr.append([res.x[0], res.x[1], res.fun])
        print(res.x, res.fun)
    top_arr = np.array(top_arr)
    ind_min = np.argmin(top_arr[:,2])
    print(f"Estimate = {top_arr[ind_min, :2]}, {j+1}/{num_simulations}")
    results.append(top_arr[ind_min, :2])
results = np.array(results)
est = pd.DataFrame(results, columns=['theta', 'phi'])
est.to_csv(f"./data/{tag}.csv", index=False, header=False)
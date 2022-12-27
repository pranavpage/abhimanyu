import numpy as np
import scipy.constants as sc
R_e = 6.3781e6
lat_e = 30
long_e = 40
lat_s = lat_e - 10
long_s = long_e + 5
f_c = 200e6
h = 500e3
R_s = R_e + h
G = sc.gravitational_constant
M = 5.9722e24
v_s = np.sqrt(G*M/(R_e+h))
print(f"Altitude of satellite : {h/1e3:.1f} km \nVelocity = {v_s*3600/1e3:.1f} kmph")
max_slant_angle = np.arcsin(R_e/(R_e+h))
max_slant_range = R_s*np.cos(max_slant_angle)
print(f"Max slant angle : {np.degrees(max_slant_angle):.2f} degrees")
print(f"Center Frequency = {f_c/1e6:.2f} MHz")
print(f"Max slant range : {max_slant_range/1e3:.2f} km")
print(f"Footprint area : {2*np.pi*(R_e**2)*(1-np.cos(np.pi/2 - max_slant_angle))/1e6:.2f} sq km")
def cart(s_pos):
    return np.array([s_pos[0]*np.sin(s_pos[1])*np.cos(s_pos[2]), s_pos[0]*np.sin(s_pos[1])*np.sin(s_pos[2]), s_pos[0]*np.cos(s_pos[1])])
polar_e = [R_e, np.radians(90-lat_e), np.radians(long_e)]
polar_s = [R_s, np.radians(90-lat_s), np.radians(long_s)]
polar_vpoint = [R_s/np.cos(polar_s[1]), 0, 0]
cart_s = cart(polar_s)
cart_e = cart(polar_e)
cart_vpoint = cart(polar_vpoint)
vec1 = cart_vpoint - cart_s
vec2 = cart_e - cart_s
dot = np.dot(vec1, vec2)/(np.linalg.norm(vec1, 2)*np.linalg.norm(vec2, 2))
doppler_angle = np.arccos(dot)
print(f"Doppler Angle : {np.degrees(doppler_angle):.2f} degrees")
print(f"Distance between emitter and satellite : {np.linalg.norm(vec2, 2)/1e3:.2f} km")
print(f"Relative velocity : {v_s*np.cos(doppler_angle)*3600/1e3:.2f} kmph")
doppler_shift = v_s*np.cos(doppler_angle)*f_c/sc.c
print(f"Doppler shift : {doppler_shift:.2f} Hz")

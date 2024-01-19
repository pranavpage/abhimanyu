import numpy as np
altitude=500e3 
Re = 6.371e6
angle_traversed = 2*np.arccos(Re/(Re + altitude))
orbital_period=97*60 
traversal_time = angle_traversed/(2*np.pi/orbital_period)
print(f"altitude = {altitude/1e3:.2f} km, traversal time = {traversal_time/60:.3f} minutes")
flyby= 6*60 
print(f"Flyby duration = {flyby/60:.3f} minutes")
bandwidth=50e6
sampling_rate = 2*bandwidth
print(f"Considering raw I/Q data storage and dump")
print(f"bandwidth = {bandwidth/1e6:.3f} MHz, sampling rate = {sampling_rate/1e6:.3f} M Samples/sec")
num_bits_per_sample_per_channel = 16 
num_bits_per_sample = num_bits_per_sample_per_channel*2
size_of_file = flyby*sampling_rate*num_bits_per_sample
print(f"Num samples = {flyby*sampling_rate:.2e} ")
print(f"Data arrival rate = {num_bits_per_sample*sampling_rate/1e6:.3f} Mbits/s")
print(f"Size of file = {size_of_file:.3e} bits, {(size_of_file/8)/(1e9):.3f} GB")
downlink_rate = 500e6
downlink_time = size_of_file/downlink_rate
print(f"Downlink time = {downlink_time/60:.3f} minutes\n")


print(f"Storing PDWs and downlink")
num_float_fields = 7
num_int_fields = 1
data_bits = 8*(num_int_fields + 1) + 32*(num_float_fields+3)
print(f"Payload for one PDW = {data_bits} bits or {data_bits/8} bytes")
pulse_arrival_rate = 1e6 
print(f"{pulse_arrival_rate/1e6:.2f} million pulses arriving at satellite per second")
file_size_pdw = pulse_arrival_rate*flyby*data_bits
print(f"File size = {file_size_pdw/8e9:.2f} GB")
pdw_data_gen_rate = pulse_arrival_rate*data_bits
print(f"Data arrival rate = {pdw_data_gen_rate/1e6:.3f} Mbits/s")
print(f"Considering available downlink rate = {downlink_rate/1e6:.2f} Mbits/s")



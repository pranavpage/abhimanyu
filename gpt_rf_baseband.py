import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
import pickle
def generate_rf_complex_baseband_signal(prf, pulse_width, rise_time, fall_time, duty_cycle, num_samples, chirp_bandwidth, sampling_rate, chirp_start_freq):
    # Calculate pulse repetition interval (PRI) from PRF
    pri = 1 / prf

    # Calculate pulse length (time the pulse is ON)
    pulse_length = pulse_width + rise_time + fall_time

    # Calculate the time axis
    time = np.linspace(0, num_samples / sampling_rate, num_samples)

    # Calculate the number of samples in one PRI
    samples_per_pri = int(np.ceil(pri * sampling_rate))
    chirp_num_samples = int(duty_cycle * samples_per_pri)
    idle_num_samples = samples_per_pri - chirp_num_samples
    f_offset = 0e6
    chirp_time = np.linspace(0, pulse_length, chirp_num_samples)
    chirp_frequency = np.linspace(chirp_start_freq,chirp_start_freq+chirp_bandwidth, chirp_num_samples)
    chirp_waveform = np.exp(1j * 2 * np.pi * (chirp_frequency) * chirp_time)*np.exp(1j * 2 * np.pi * (f_offset) * chirp_time)  # Complex chirp waveform

    # Add rise and fall time to the chirp waveform
    rise_samples = int(rise_time * sampling_rate)
    fall_samples = int(fall_time * sampling_rate)
    chirp_waveform[:rise_samples] *= np.linspace(0, 1, rise_samples)
    chirp_waveform[-fall_samples:] *= np.linspace(1, 0, fall_samples)

    # Initialize the complex baseband signal with all zeros
    complex_baseband_signal = np.zeros(num_samples, dtype=complex)

    # Calculate the number of pulses based on duty cycle and PRI
    num_pulses = int(num_samples // samples_per_pri)

    # Insert the complex chirp pulses into the complex baseband signal with the given PRF
    for i in range(num_pulses):
        complex_baseband_signal[i * samples_per_pri : i * samples_per_pri + chirp_num_samples] = chirp_waveform

    return time, complex_baseband_signal

def add_white_gaussian_noise(signal, snr_db, gaussian_noise_power):
    # Calculate the power of the original signal
    power_signal = np.abs(signal)**2
    signal_power = power_signal[np.nonzero(power_signal)].mean()
    # we want scaled signal power / noise power  = snr linear
    # sscaling factor = (snr_linear * noise_power)/(signal_power)
    # Calculate the noise power to achieve the desired SNR
    snr_linear = 10 ** (snr_db / 10.0)


    # noise_power = signal_power / (2 * snr_linear)  # Divide by 2 for I and Q channels
    noise_power = gaussian_noise_power/2
    scaling_factor = (snr_linear * 2 * noise_power)/signal_power
    # Generate white Gaussian noise for I and Q channels separately
    noise_i = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noise_q = np.random.normal(0, np.sqrt(noise_power), len(signal))

    # Add noise to the original signal in I and Q channels
    noisy_signal = np.sqrt(scaling_factor)*signal + noise_i + 1j * noise_q

    return noisy_signal

def generate_noisy_rf_complex_baseband_signal(prf, pulse_width, rise_time, fall_time, duty_cycle, num_samples, chirp_bandwidth, sampling_rate, snr_db, start_time, gaussian_noise_power, chirp_start_freq):
    # Generate the RF complex baseband signal with chirp waveform
    time, complex_baseband_signal = generate_rf_complex_baseband_signal(prf, pulse_width, rise_time, fall_time, duty_cycle, num_samples, chirp_bandwidth, sampling_rate, chirp_start_freq)

    # Add white Gaussian noise to the signal with the specified SNR
    noisy_complex_baseband_signal = add_white_gaussian_noise(complex_baseband_signal, snr_db, gaussian_noise_power)

    # Shift the time axis to introduce the start time delay
    shifted_time = time - start_time

    # Generate complex baseband signal with random delay
    shifted_complex_baseband_signal = np.interp(time, shifted_time, noisy_complex_baseband_signal)

    return time, shifted_complex_baseband_signal

def calculate_times_of_arrival(complex_baseband_signal, threshold, sampling_rate):
    # Find indices where the amplitude of the signal crosses the threshold
    crossings = np.where(np.abs(complex_baseband_signal) > threshold)[0]

    # Calculate times of arrival in seconds for each crossing
    times_of_arrival = crossings / sampling_rate

    return times_of_arrival

# Example parameters
prf_1 = 700  # PRF in Hz (e.g., 1000 pulses per second)
rise_time_1 = 1e-6  # Rise time in seconds (e.g., 1 microsecond)
fall_time_1 = 1e-6  # Fall time in seconds (e.g., 1 microsecond)
duty_cycle_1 = 0.02  # Duty cycle (e.g., 5%)
pulse_width_1 = (1/prf_1) * duty_cycle_1  # Pulse width
num_samples = int(1e6)  # Number of samples in the baseband signal
chirp_start_freq_1 = 1e6
chirp_bandwidth_1 = 0.2e6  # Chirp bandwidth in Hz (e.g., 1 MHz)
sampling_rate = 100e6  # Sampling rate in Hz (e.g., 10 MHz)
snr_db_1 = 12  # Desired SNR in dB
gaussian_noise_power = 0.1
# Example parameters for the second pulse train
prf_2 = 1100  # PRF in Hz (e.g., 2000 pulses per second)
rise_time_2 = 0.25e-6  # Rise time in seconds (e.g., 0.5 microseconds)
fall_time_2 = 0.25e-6  # Fall time in seconds (e.g., 0.5 microseconds)
duty_cycle_2 = 0.01  # Duty cycle (e.g., 10%)
pulse_width_2 = (1/prf_2) * duty_cycle_2  # Pulse width
chirp_start_freq_2 = 0.5e6
chirp_bandwidth_2 = 2.5e6  # Chirp bandwidth in Hz (e.g., 2 MHz)
snr_db_2 = 12 # Desired SNR in dB


# Example parameters for the third radar
prf_3 = 900  # PRF in Hz
rise_time_3 = 0.5e-6  # Rise time in seconds
fall_time_3 = 0.5e-6  # Fall time in seconds
duty_cycle_3 = 0.02  # Duty cycle
pulse_width_3 = (1/prf_3) * duty_cycle_3  # Pulse width
chirp_start_freq_3 = 2e6
chirp_bandwidth_3 = 1e6  # Chirp bandwidth in Hz
snr_db_3 = 15  # Desired SNR in dB

print(f"Pulse widths = {pulse_width_1, pulse_width_2, pulse_width_3}")
# Generate random start times for each pulse train
# start_time_1 = np.random.uniform(0, 1/prf_1)
# start_time_2 = np.random.uniform(0, 1/prf_2)
# start_time_3 = np.random.uniform(0, 1/prf_3)
start_time_1 = (1/prf_1)*0.1
start_time_2 = (1/prf_2)*0.4
start_time_3 = (1/prf_3)*0.2

# Generate the noisy RF complex baseband signals with chirp waveform and desired SNR, including random delays
time_1, noisy_complex_baseband_signal_1 = generate_noisy_rf_complex_baseband_signal(prf_1, pulse_width_1, rise_time_1, fall_time_1, duty_cycle_1, num_samples, chirp_bandwidth_1, sampling_rate, snr_db_1, start_time_1, gaussian_noise_power, chirp_start_freq_1)

time_2, noisy_complex_baseband_signal_2 = generate_noisy_rf_complex_baseband_signal(prf_2, pulse_width_2, rise_time_2, fall_time_2, duty_cycle_2, num_samples, chirp_bandwidth_2, sampling_rate, snr_db_2, start_time_2, gaussian_noise_power, chirp_start_freq_2)

time_3, noisy_complex_baseband_signal_3 = generate_noisy_rf_complex_baseband_signal(prf_3, pulse_width_3, rise_time_3, fall_time_3, duty_cycle_3, num_samples, chirp_bandwidth_3, sampling_rate, snr_db_3, start_time_3, gaussian_noise_power, chirp_start_freq_3)

# Combine the three complex baseband signals
combined_complex_baseband_signal = noisy_complex_baseband_signal_1 + noisy_complex_baseband_signal_2 + noisy_complex_baseband_signal_3

threshold = 4*gaussian_noise_power  # Change this value based on the amplitude of the pulse and noise level
print(f"Threshold = {threshold}")
# Calculate times of arrival for the combined complex baseband signal

def smooth_power_envelope(signal, window_size):
    # Calculate the squared magnitude (power) of the complex signal
    power_signal = np.abs(signal) ** 2

    # Apply a moving average to smooth the power values
    window = np.ones(window_size) / window_size
    smoothed_power = np.convolve(power_signal, window, mode='same')

    return smoothed_power

# ... (Previous code remains the same)

# Calculate the smoothed power envelope for the combined complex baseband signal
window_size = 125  # Adjust the window size for smoothing (larger value for more smoothing)
smoothed_power_envelope = smooth_power_envelope(combined_complex_baseband_signal, window_size)
def detect_edges(data, thresh):
    # Determine the sign of the data compared to the threshold
    sign = data >= thresh

    # Find rising edges (transitions from False to True)
    rising_edges = np.where(np.convolve(sign, [1, -1]) == 1)[0]

    # Find falling edges (transitions from True to False)
    falling_edges = np.where(np.convolve(sign, [-1, 1]) == 1)[0]

    return rising_edges, falling_edges
rising_edges, falling_edges = detect_edges(smoothed_power_envelope, threshold)
t_start = rising_edges*1/sampling_rate
t_end = falling_edges*1/sampling_rate
def find_crossings_within_window(smoothed_power_envelope, start_idx, end_idx, fraction_of_max_power):
    # Get the max power within the specified window
    max_power_in_window = np.max(smoothed_power_envelope[start_idx:end_idx + 1])

    # Calculate the threshold power based on the fraction of max power
    threshold_power = fraction_of_max_power * max_power_in_window

    # Extract the smoothed power envelope within the window
    envelope_in_window = smoothed_power_envelope[start_idx:end_idx + 1]

    # Determine the sign of the data compared to the threshold
    sign = envelope_in_window >= threshold_power

    # Find rising edges (transitions from False to True)
    rising_edges = np.where(np.convolve(sign, [1, -1]) == 1)[0]

    # Find falling edges (transitions from True to False)
    falling_edges = np.where(np.convolve(sign, [-1, 1]) == 1)[0]

    # Shift the rising and falling edge indices to account for the start index of the window
    rising_edges += start_idx
    falling_edges += start_idx

    return [rising_edges[0], falling_edges[0]]
half_crossings = np.array([find_crossings_within_window(smoothed_power_envelope, rising_edges[i], falling_edges[i], 0.5) for i in range(len(rising_edges))])
start_idx = half_crossings[:,0]
end_idx = half_crossings[:,1]

t_start = half_crossings[:, 0]*1/sampling_rate
t_end = half_crossings[:, 1]*1/sampling_rate

low_crossings = np.array([find_crossings_within_window(smoothed_power_envelope, rising_edges[i], falling_edges[i], 0.1) for i in range(len(rising_edges))])
rise_start_idx = low_crossings[:,0]
fall_end_idx = low_crossings[:, 1]
t_rise_start = low_crossings[:, 0]/sampling_rate
t_fall_end = low_crossings[:, 1]/sampling_rate

high_crossings = np.array([find_crossings_within_window(smoothed_power_envelope, rising_edges[i], falling_edges[i], 0.8) for i in range(len(rising_edges))])
t_rise_end = high_crossings[:, 0]/sampling_rate
t_fall_start = high_crossings[:, 1]/sampling_rate
pulse_max_power = np.array([max(smoothed_power_envelope[start_idx[i]:end_idx[i]+1]) for i in range(len(start_idx))])
pulse_raw = [combined_complex_baseband_signal[rise_start_idx[i]:fall_end_idx[i]+1] for i in range(len(rise_start_idx))]
with open("pulses_raw", "wb") as fp:
    pickle.dump(pulse_raw, fp)
# with open("pulses_raw", "rb") as fp:
#     pulse_pickled = pickle.load(fp)
#     print(pulse_pickled[0])

# change start end points of raw pulse 
t_rise = t_rise_end - t_rise_start
t_fall = t_fall_end - t_fall_start
t_arrival = t_rise_start
pulse_width = t_end - t_start
# pulse_width = t_fall_end - t_rise_start


def calculate_frequency_spectrum(signal, sampling_rate):
    # Calculate the length of the signal
    signal_length = len(signal)

    # Calculate the frequency resolution
    frequency_resolution = sampling_rate / signal_length

    # Compute the FFT of the signal
    frequency_spectrum = np.fft.fft(signal)

    # Shift the zero-frequency component to the center
    frequency_spectrum = np.fft.fftshift(frequency_spectrum)

    # Calculate the corresponding frequency axis
    frequencies = np.fft.fftfreq(signal_length, d=1/sampling_rate)
    frequencies = np.fft.fftshift(frequencies)

    return frequencies, np.abs(frequency_spectrum)
def calculate_centroid_frequency(frequencies, spectrum):
    # Calculate the weighted average of frequencies using the spectrum as weights
    centroid_frequency = np.sum(frequencies * spectrum) / np.sum(spectrum)
    return centroid_frequency
def calculate_complex_baseband_bandwidth(signal, sampling_rate):
    # Compute the power spectral density (PSD) using the Fourier Transform
    fft_result = np.fft.fft(signal)
    psd = np.abs(fft_result)**2

    # Calculate the frequency resolution (bin size)
    frequency_resolution = sampling_rate / len(signal)

    # Create the frequency axis for the PSD
    frequencies = np.fft.fftfreq(len(signal), d=1/sampling_rate)

    # Find the positive frequency bins in the PSD
    positive_bins = frequencies >= 0
    psd_positive = psd[positive_bins]
    frequencies_positive = frequencies[positive_bins]

    # Find the maximum frequency bin in the positive spectrum
    max_bin_idx = np.argmax(psd_positive)

    # Calculate the bandwidth as the frequency resolution times the index of the max bin
    bandwidth = frequency_resolution * max_bin_idx

    return bandwidth
pulse_bandwidth = [calculate_complex_baseband_bandwidth(pulse_raw[i], sampling_rate) for i in range(len(start_idx))]

centroid_frequencies = []
for i in range(len(start_idx)):
    # Calculate the frequency spectrum of the pulse
    frequencies, spectrum = calculate_frequency_spectrum(pulse_raw[i], sampling_rate)

    # Calculate the centroid frequency
    centroid_frequency = calculate_centroid_frequency(frequencies, spectrum)
    centroid_frequencies.append(centroid_frequency)
# centroid_frequencies = [calculate_centroid_frequency(calculate_frequency_spectrum(pulse_raw[i], sampling_rate)) for i in range(len(pulse_raw))]
pdw_table = pd.DataFrame({"t_arrival":t_arrival, "t_rise":t_rise, "t_fall":t_fall, "pulse_width":pulse_width , "pulse_max_power":pulse_max_power, "pulse_bandwidth":pulse_bandwidth, "pulse_center_freq":centroid_frequencies})
print(pdw_table)
pdw_table.to_csv(f"pdw_table.csv", index=False)
# plt.figure(2)
# plt.scatter(pdw_table['pulse_width']*1e6, pdw_table['pulse_center_freq']/1e6)
# plt.xlabel("Pulse width (µs)")
# plt.ylabel("Pulse center freq (MHz) ")
# plt.grid()
# plt.show()
# Get intervals where smoothened power envelope is above threshold, find max power in those intervals, find time when smoothened power rises to fraction of max power and falls below fraction of max power
# Plot the original combined complex baseband signal and the smoothed power envelope
plt.figure(1)
plt.plot(time_1 * 1e6, combined_complex_baseband_signal.real, '-b', label='I', alpha=0.7)
plt.plot(time_1 * 1e6, combined_complex_baseband_signal.imag, '-r', label='Q', alpha=0.7)
# plt.plot(time_1 * 1e6, smoothed_power_envelope, '-k', label='Smoothed Power Envelope')
plt.xlabel('Time (µs)')
# plt.xlim(t_start[1]*1e6 - 10, t_end[1]*1e6 + 10)
plt.ylabel('Amplitude')
plt.title('Pulse Parameters Extracted')
plt.legend()
plt.grid(True)
# plt.show()
# plt.axhline(threshold, linestyle = "-.", alpha=0.5, label="Threshold")
for arrival_time in t_start:
    plt.axvline(x=arrival_time * 1e6, color='g', linestyle='--', alpha=0.7)
for arrival_time in t_end:
    plt.axvline(x=arrival_time * 1e6, color='m', linestyle='--', alpha=0.7)
# for arrival_time in t_end:
#     plt.axvline(x=arrival_time * 1e6, color='m', linestyle='--', alpha=0.7)

# plt.grid(True)
# plt.legend()
plt.show()
# plt.savefig("./plots/ppe_pulse_td.png")

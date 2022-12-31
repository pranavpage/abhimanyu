import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt
data = np.fromfile("bpsk_test.dat", dtype = np.float32)
f, pxx = sp.periodogram(data, 500e6)
plt.figure(0)
plt.plot(f, pxx)
plt.show()
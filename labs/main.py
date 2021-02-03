from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
from chirp.base import *

bw = 4e6
tp = 10e-6
fs = 8*bw
ts = 1/fs
n = fs*tp
t = np.arange(n)*ts

x = signal.chirp(t, 0e6, tp, 1e6)
y = signal.chirp(t, 1e6, tp, 2e6)
plt.plot(t, x, label='f0=0MHz, f1=1MHz')
plt.plot(t, y, label='f0=1MHz, f1=2MHz')
plt.legend()
plt.grid()
plt.show()

c, lags = xcorr(x, y)
plt.plot(lags*ts, mag2db(c))
plt.ylim([-50, 0])
plt.grid()
plt.show()

f, pxx = signal.periodogram([x, y], fs=fs, nfft=4096, return_onesided=False, scaling='spectrum')
plt.plot(f, mag2db(pxx[0]), label='Pxx')
plt.plot(f, mag2db(pxx[1]), label='Pyy')
plt.xlim([-4e6, 4e6])
plt.ylim([-50, 0])
plt.grid()
plt.show()

f, t, sxx = signal.spectrogram(y, fs=fs, nperseg=160, noverlap=159, nfft=512, scaling='spectrum')
plt.pcolormesh(t, f, sxx, shading='gouraud', cmap='gray_r')
plt.ylim([0, 4e6])
plt.colorbar()
plt.show()

#%%
nyq = 0.5*fs
low, high = 1e6*2/nyq, 3e6/nyq
b, a = signal.butter(5, [low, high], btype='band')

t = np.arange(n)*ts
plt.plot(t, y)
plt.plot(t, signal.filtfilt(b, a, x))
plt.show()

#%%
c, lags = xcorr(y, signal.filtfilt(b, a, x))
plt.plot(lags*ts, mag2db(c))
# plt.ylim([-50, 0])
plt.grid()
plt.show()

#%%
f, pxx = signal.periodogram([y, signal.filtfilt(b, a, x)], fs=fs, nfft=4096, return_onesided=False, scaling='spectrum')
plt.plot(f, mag2db(pxx[0]), label='Pxx')
plt.plot(f, mag2db(pxx[1]), label='Pyy')
plt.show()

#%%
fx = signal.filtfilt(b, a, x)
plt.plot(fx)
plt.plot(fx/np.max(fx))
plt.plot(y)
plt.show()
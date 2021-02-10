# auto and cross-correlation level
# compare 4MHz signal for 1us

from chirp.base import *
from matplotlib import pylab as plt

b, tp, oversample = 16e6, 1e-6, 32
x = chirp(b, tp, oversample, (0, 4e6))
y = chirp(b, tp, oversample, (4e6, 8e6))
z = chirp(b, tp, oversample, (4e6, 0))

rxy = xcorr(x, y, lags=False)
rxz, lags = xcorr(x, z)

plt.plot(lags, rxy, label='(0, 4MHz), (4MHz, 8MHz)')
plt.plot(lags, rxz, label='(0, 4MHz), (4MHz, 0MHz)')
plt.legend()
plt.grid(alpha=0.5, linestyle='--')
plt.show()

#%%
b, tp, oversample = 64e6, 1e-6, 2
x = chirp(b, tp, oversample, (0, 16e6))
y = chirp(b, tp, oversample, (16e6, 0e6))

rxx = xcorr(x, x, lags=False)
rxy, lags = xcorr(x, y)
rxx[len(rxx)//2] = 0

plt.plot(lags, rxx, label='autocorrelation sidelobe')
plt.plot(lags, rxy, label='cross-correlation')
plt.grid(alpha=0.5, linestyle='--')
plt.vlines(lags[np.argmax(rxy)], np.min(rxx), np.max(rxy), linestyles='--', colors='r')
plt.legend()
plt.show()

print(np.sum(np.abs(rxx) ** 2), np.max(np.abs(rxx)))
print(np.sum(np.abs(rxy) ** 2), np.max(np.abs(rxy)))
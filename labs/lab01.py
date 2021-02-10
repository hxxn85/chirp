from chirp.base import *
from matplotlib import pyplot as plt

f0, f1 = freqx(32e6, 4)
x = chirp(32e6, 1e-6, 8, (f0[3], f1[3]))
rxx, lags = xcorr(x, x)

plt.plot(lags[len(lags)//2+1:], np.abs(rxx[len(lags)//2+1:]))
plt.plot()
plt.show()
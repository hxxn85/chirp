from chirp.base import *
from matplotlib import pyplot as plt

f0, f1 = freqx(32e6, 4)
x = chirp(32e6, 1e-6, 8, (f0[0], f1[0]))
rxx, lags = xcorr(x, x)

plt.plot(lags, rxx)
plt.show()
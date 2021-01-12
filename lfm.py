from matplotlib import pyplot as plt
from gwu import *

b = 4e6
n = 65536
tp, res, w = np.array([10, 20, 40, 80, 160, 320])*1e-6, [], []

for i in range(len(tp)):
    fs = n/tp[i]
    ts = 1/fs
    t = np.arange(n)/n*tp[i]
    x = signal.chirp(t, 0, tp[i], b)
    f, pxx = signal.periodogram(x, fs, nfft=65536*2, return_onesided=False, scaling='spectrum')
    w.append(np.array(f))
    res.append(mag2db(pxx))

[plt.plot(f*1e-6, pxx, label=f'TB={int(b*t)}, B=4MHz, Tp={int(t*1e6)}$\mu$s') for f, pxx, t in zip(w, res, tp)]
plt.xlim([-2*b*1e-6, 2*b*1e-6])
plt.ylim([-100, 10])
plt.xlabel('Frequency (MHz)')
plt.ylabel('$|X(f)|^2$')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.show()

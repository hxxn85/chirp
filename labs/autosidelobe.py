from chirp.base import *
from matplotlib import pyplot as plt

f0, f1 = freqx(32e6, 4)
x = chirp(32e6, 1e-6, 8, (f0[0], f1[0]))
rxx, lags = xcorr([1, 1, 1, 1], [1, 1, 1, 1])

plt.plot(lags, rxx)
plt.show()

#%%
from scipy import linalg
import numpy as np

h = linalg.hadamard(4)

def sq2tri(x):
    queue, y = [], []
    for e in x:
        if len(queue) == 0:
            queue.append(e)
        elif queue[-1] == e:
            queue.append(e)
        else:
            k = len(queue)
            v = queue[-1]
            y += list(np.linspace(v, -v, k, endpoint=False))
            queue.clear()
            queue.append(e)

    k = len(queue)
    v = queue[-1]
    y += list(np.linspace(v, -v, k, endpoint=False))
    return y

idx = 0
a = h[idx].repeat(32)
b = sq2tri(a)

plt.figure()
plt.subplot(2,1,1)
plt.title(f'index={idx}')
plt.plot(a, label='square')
plt.plot(b, label='triangle')
plt.grid(alpha=0.5, linestyle='--')
plt.legend()

plt.subplot(2,1,2)
raa, _ = xcorr(a, a)
rbb, lags = xcorr(b, b)
plt.plot(lags, raa/np.max(raa), label='square')
plt.plot(lags, rbb/np.max(rbb), label='triangle')
plt.title('Autocorrelation')
plt.grid(alpha=0.5, linestyle='--')
plt.legend()

plt.tight_layout()
plt.show()

#%%
import itertools
comb = list(itertools.combinations(np.arange(4), 2))
plt.figure()
for p, k in zip(comb, np.arange(len(comb))):
    plt.subplot(2,3,k+1)
    i, j = p
    x, y = h[i], h[j]
    rxy, lags = xcorr(x, y)
    plt.plot(lags, rxy, label='square')

    rxy, lags = xcorr(sq2tri(x), sq2tri(y))
    plt.plot(lags, rxy, label='triangle')

    plt.title(f'i={i}, j={j}')
    plt.xlabel('lags')
    plt.ylabel('cross-correlation')
    plt.grid(alpha=0.5, linestyle='--')

plt.tight_layout()
plt.show()

#%%
b, tp = 4e6, 1e-6
fs = 8*b
n = fs*tp
idx = 1
h = linalg.hadamard(32)
x = h[idx]
y = sq2tri(x)

f, pxx = signal.periodogram(x, fs, nfft=1024)
_, pyy = signal.periodogram(y, fs, nfft=1024)
plt.plot(f/1e6, pxx)
plt.plot(f/1e6, pyy)
plt.title(f'index={idx}')
plt.xlabel('frequency (MHz)')
plt.ylabel('amplitude')
plt.grid(alpha=0.5, linestyle='--')
plt.show()
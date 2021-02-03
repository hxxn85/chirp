from chirp.base import *
from matplotlib import pyplot as plt
import itertools
import numpy as np

def plot_selected(b, tp, nseg, f):
    f0, f1 = freqx(b, nseg)
    freq = list(zip(f0, f1))
    [plt.plot([0, tp * 1e6], freq[i], color='k', alpha=0.5) for i in range(len(freq))]
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlabel('Time ($\mu s$)')
    plt.ylabel('Frequency (Hz)')

    [plt.plot([0, tp * 1e6], y, color='r') for y in f]
    plt.show()

def plot_correlation(b, tp, oversamplerate, f, scale='normal'):
    comb = list(itertools.combinations(np.arange(len(f)), 2))
    peaks = []
    for i, j in comb:
        x = chirp(b, tp, oversamplerate, f[i])
        y = chirp(b, tp, oversamplerate, f[j])

        rxy, lags = xcorr(x, y)
        if scale == 'normal':
            print(f'{f[i]}, {f[j]}, {np.max(np.abs(rxy))}')
            plt.plot(lags, rxy, label=f'{f[i]}, {f[j]}')
            peaks.append(np.max(np.abs(rxy)))
        elif scale == 'db':
            rxy = pow2db(np.abs(rxy))
            print(f'{f[i]}, {f[j]}, {np.max(rxy)}')
            plt.plot(lags, rxy, label=f'{f[i]}, {f[j]}')
            peaks.append(np.max(rxy))

    print(f'max={np.max(peaks)}, min={np.min(peaks)}, mean={np.mean(peaks)}, std={np.std(peaks)}')
    plt.legend()
    plt.grid(alpha=0.5, linestyle='--')
    plt.show()

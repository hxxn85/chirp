import numpy as np
from scipy import signal

def freqx(b: float, nradar: int) -> tuple:
    nseg = 2**np.floor(np.log2(nradar))
    f0, f1 = [], []
    while nseg >= 1:
        x = list(np.linspace(0, b, int(nseg), endpoint=False))
        f0.append(x)
        f1.append(x + b/nseg)
        nseg *= 0.5
    return [x for y in f0 for x in y], [x for y in f1 for x in y]

def mag2db(x):
    return 20 * np.log10(abs(np.array(x)))

def pow2db(x):
    return 10 * np.log10(abs(np.array(x)))

def xcorr(x, y, normalize=True, lags=True):
    a, b = np.array(x), np.array(y)
    if normalize:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / np.std(b)

    if lags:
        return signal.correlate(a, b), signal.correlation_lags(len(x), len(y))
    else:
        return signal.correlate(a, b)
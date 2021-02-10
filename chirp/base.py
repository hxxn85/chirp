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

    a, b = [x for y in f0 for x in y], [x for y in f1 for x in y]
    return a + b, b + a

def mag2db(x):
    return 20 * np.log10(abs(np.array(x)))

def pow2db(x):
    return 10 * np.log10(abs(np.array(x)))

def xcorr(x, y, normalize=True, lags=True):
    rxx0 = np.max(signal.correlate(x, x))
    ryy0 = np.max(signal.correlate(y, y))
    rxy = signal.correlate(x, y)
    if normalize:
        rxy = rxy / np.sqrt(rxx0 * ryy0)

    if lags:
        return rxy, signal.correlation_lags(len(x), len(y))
    else:
        return rxy

def chirp(b, tp, oversamplerate, f):
    fs = oversamplerate * b
    n = int(fs*tp)
    t = np.arange(n)/fs

    return signal.chirp(t, f[0], tp, f[1])

def calc_auto(x, method='sum'):
    if method not in ['sum', 'max']:
        raise ValueError("method should be 'sum' or 'max'")

    rxx = xcorr(x, x, lags=False)
    rxx[len(rxx)//2] = 0
    if method == 'sum':
        return np.sum(np.abs(rxx) ** 2)
    elif method == 'max':
        return np.max(np.abs(rxx))

def calc_cross(x, y, method='sum'):
    if method not in ['sum', 'max']:
        raise ValueError("method should be 'sum' or 'max'")

    rxy = xcorr(x, y, lags=False)
    if method == 'sum':
        return np.sum(np.abs(rxy) ** 2)
    elif method == 'max':
        return np.max(np.abs(rxy))
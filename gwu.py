import numpy as np
from scipy import signal, special

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
    return 20 * np.log10(abs(x))

def xcorr(x, y, normalize=True):
    a, b = np.array(x), np.array(y)
    if normalize:
        a = (a - np.mean(a)) / (np.std(a) * len(a))
        b = (b - np.mean(b)) / np.std(b)
    return signal.correlate(a, b), signal.correlation_lags(len(x), len(y))

class Radar:
    def __init__(self, bw: float, tp:float, f0: float, f1: float):
        self.fs = 8 * bw
        self.ts = 1 / self.fs
        n = self.fs * tp
        self.t = np.arange(n)*self.ts
        self.x = signal.chirp(self.t, f0, tp, f1)

        nyq = 0.5 * self.fs
        if f0 == 0:
            self._b, self._a = signal.butter(5, f1 / nyq, btype='low')
        else :
            self._b, self._a = signal.butter(5, [f0 / nyq, f1 / nyq], btype='band')

    def evaluate(self, y, mode='full'):
        # fy = signal.filtfilt(self._b, self._a, y)
        # fy = signal.lfilter(self.x[::-1], 1, y)
        fy = y

        [c, lags] = xcorr(self.x, fy)
        if mode == 'max':
            return np.max(c)
        else:
            return c, lags*self.ts
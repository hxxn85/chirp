import numpy as np
from scipy import signal

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
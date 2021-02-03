from scipy.optimize import dual_annealing
from chirp.base import *
import itertools
import numpy as np

def _is_unique(x):
    x = np.array(x).astype(np.int32)
    return len(np.unique(x)) == len(x)

def _chirp_correlate(idx1, idx2, *args):
    b, tp, nseg, f0, f1 = args
    fs = 8*b
    t = np.arange(int(fs*tp))/fs

    x = signal.chirp(t, f0[idx1], tp, f1[idx1])
    y = signal.chirp(t, f0[idx2], tp, f1[idx2])
    return  np.max(np.abs(xcorr(x, y, lags=False)))

def func(x, *args):
    """
    objective(cost) function for simulated annealing
    max cross correlation peak over available set of chirp signals
    :param x: 1-D array, parameter
    :param args: b, tp, nseg
    :return: f(x)
    """
    x = np.array(x).astype(np.int32)
    if not _is_unique(x):
        return 1000

    comb = list(itertools.combinations(x, 2))
    return np.sum([_chirp_correlate(f[0], f[1], *args) for f in comb])

def callback(x, f, context):
    print(np.array(x).astype(np.int32), f, context)

def simulated_annealing(b, tp, nseg, nradar, objfun, callback=None):
    """
    :param b: bandwidth
    :param tp: pulse width
    :param nseg: number of bandwidth segmentation
    :param nradar: number of radar
    :param objfun: objective function
    :param callback: callback function, (x, f, context) will be passed to the callback function
    :return: optimize result
    """

    print('simulated annealing is running')
    nseg = int(2 ** np.ceil(np.log2(nseg)))
    print(f'b={b}, tp={tp}, nseg={nseg}, nradar={nradar}')
    print(f'objective function={objfun.__name__}')
    print(f'callback={callback.__name__}')
    f0, f1 = freqx(b, nseg)
    lower, upper = 0, len(f0)
    bound = list(zip([lower]*nradar, [upper]*nradar))
    ret =  dual_annealing(objfun, bound, (b, tp, nseg, f0, f1), callback=callback)
    x = np.array(ret.x).astype(np.int32)
    return [(f0[i], f1[i]) for i in x], ret.fun

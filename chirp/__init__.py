from chirp import optimizer as op
import numpy as np

ret = op.simulated_annealing(32e6, 1e-6, 16, 2, op.func, op.callback)
ret.x = ret.x.astype(np.int32)
print(sorted(ret.x), ret.fun)
from chirp import optimizer as op
from chirp import visualizer

b, tp, nseg, nradar = 32e6, 1e-6, 32, 4
x, fun = op.simulated_annealing(b, tp, nseg, nradar, op.func, op.callback)
x = sorted(x)
print(x, fun)

visualizer.plot_selected(b, tp, nseg, x)
visualizer.plot_correlation(b, tp, 8, x)

from matplotlib import pyplot as plt
from scipy import optimize
from gwu import *

b, tp, N = 4e6, 10e-6, 7
f0, f1 = freqx(b, N)
# generate N radars
radar = [Radar(b, tp, f0[i], f1[i]) for i in range(N)]

#%% spectrum of two adjacent radars
[f, pxx] = signal.periodogram([radar[0].x, radar[1].x], fs=radar[0].fs, nfft=65536, return_onesided=False,
                              scaling='spectrum')
plt.plot(f, mag2db(pxx[0]), label=f'radar[0], f0={f0[0]/1e6}, f1={f1[0]/1e6}')
plt.plot(f, mag2db(pxx[1]), label=f'radar[1], f0={f0[1]/1e6}, f1={f1[1]/1e6}')
plt.grid()
plt.ylim([-100, 0])
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.show()

#%% x-correlation effect
[c, lags] = xcorr(radar[0].x, radar[1].x)
plt.plot(lags*radar[0].ts*1e6, mag2db(c))
plt.grid()
plt.xlabel('time ($\mu$s)')
plt.ylabel('$R_{xy}(t)$')
plt.show()

#%% time delay effect
ndelay = int((np.argmax(c) - 1) / 2)
plt.plot(np.concatenate([radar[0].x, -np.ones(ndelay)]), label='x(t)')
plt.plot(np.concatenate([-np.ones(ndelay), radar[1].x]), label=f'y(t-{ndelay*radar[1].ts})')
plt.legend()
plt.show()

#%%
res = [np.round(mag2db(radar[0].evaluate(radar[k].x, mode='max')),3) for k in range(N)]
print(res)
ticks, labels = list(range(N)), [f'(0, {i})' for i in range(N)]
plt.plot(res, '-o', label=f'radar[0], f0={f0[0]/1e6}, f1={f1[0]/1e6}')
plt.legend()
plt.xticks(ticks, zip(np.array(f0)/1e6, np.array(f1)/1e6), rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

#%% curve fitting
func = lambda x, a, b: -a * np.log(x) + b
xdata = np.array([4, 8, 40, 64, 400])
ydata = np.array([-16.898, -19.927, -26.465, -28.537, -36.395])
popt, pcov = optimize.curve_fit(func, xdata, ydata)
popt = np.round(popt, 3)
print(popt)
plt.semilogx(xdata, ydata, '-o', label='data')
plt.semilogx(xdata, func(xdata, *popt), '-x', label=f'fit: a={popt[0]}, b={popt[1]}', alpha=0.8)
plt.legend()
plt.grid(alpha=0.5, which='both', linestyle='--')
plt.show()

# res = []
# for i in range(n):
#     res.append([np.round(mag2db(radar[i].evaluate(radar[k].x, mode='max')),3) for k in range(n)])
#
# plt.plot(res[0], '-o')
# plt.grid()
# plt.show()
#
# m = np.array(res)[np.triu_indices(7)]
# print(len(m[m<-20]))
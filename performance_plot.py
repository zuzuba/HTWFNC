import numpy as np

import matplotlib.pyplot as plt

dim, performance = np.loadtxt("naive_perf.dat", unpack=True)
peak = 4

#TODO: measure or set by hand peak performance

fig=plt.figure()
ax = plt.subplot(111)
plt.plot(dim, performance,'bd')
plt.plot(dim, performance,'b')
plt.plot(dim, peak*np.ones(dim.shape),'r')
plt.xlim(dim[0], dim[-1])
plt.yscale("log")
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of quantize VS matrix dimension")
plt.ylim(1e-3,peak+1)
ax.text(dim[-3],11/10*performance[-1],'naive',color='b')
ax.text(dim[-5],7/10*peak,'scalar peak performance',color='r')
plt.show()
fig.savefig('Performance_q.eps', format='eps')
fig.savefig('Performance_q.png', format='png', dpi=200)

"""
dim, cycles = np.loadtxt("runtime_deq.txt", unpack=True)
performance = 2*dim**2/cycles
peak = 4

fig=plt.figure()
ax = plt.subplot(111)
plt.plot(dim, performance,'bd')
plt.plot(dim, performance,'b')
plt.plot(dim, peak*np.ones(dim.shape),'r')
plt.xlim(dim[0], dim[-1])
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of dequantize VS matrix dimension")
plt.ylim(0,peak+1)
plt.show()
fig.savefig('Performance_deq.eps', format='eps')
"""

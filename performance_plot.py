import numpy as np

import matplotlib.pyplot as plt

dim, cycles = np.loadtxt("runtime_q.dat", unpack=True)
performance = 3*dim**2/cycles
peak = 4
percentage = performance/peak*100

fig=plt.figure()
ax = plt.subplot(111)
plt.plot(dim, performance,'bd')
plt.plot(dim, performance,'b')
plt.xlim(100, 100)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of quantize VS matrix dimension")
plt.ylim(0,1)
plt.show()
fig.savefig('Performance_q.eps', format='eps')

dim, cycles = np.loadtxt("runtime_deq.dat", unpack=True)
performance = 2*dim**2/cycles
peak = 4
percentage = performance/peak*100

fig=plt.figure()
ax = plt.subplot(111)
plt.plot(dim, performance,'bd')
plt.plot(dim, performance,'b')
plt.xlim(100, 1000)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of dequantize VS matrix dimension")
plt.ylim(0,1)
plt.show()
fig.savefig('Performance_deq.eps', format='eps')
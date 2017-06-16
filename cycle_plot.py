import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_folder = './data'
plot_folder = './plots'
peak = 4

############## QMM NAVIVE ###############################################

# Qmm_naive = qmm_kernel_naive + round_saturation_naive

fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
ax.set_ylabel('[cycles]', rotation=0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Runtime of naive qmm VS matrix dimension")
# plt.ylim(1e-3,peak+1)

number_func = 2
color = cm.rainbow(np.linspace(0, 1, number_func))

# kernel
file = 'cycles_qmm_kernel_naive.dat'
dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = np.copy(performance)
c = color[0]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1.1*cum_perf[-1], 'kernel_naive', color=c)

# round_sat
file = 'cycles_round_saturation_naive.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[1]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1.1*cum_perf[-1], 'round/sat naive', color=c)

plt.show()

fig.savefig(os.path.join(plot_folder, 'Cycles_naive.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Cycles_naive.png'), format='png',
            dpi=200)


############## QMM TRICK ###############################################

# Qmm_naive_trick = trick_vector_naive +
#                   qmm_kernel_trick +
#                   add_trick_vector_naive +
#                   round_saturation_naive

fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
ax.set_ylabel('[cycles]', rotation=0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Runtime of qmm with trick VS matrix dimension")
# plt.ylim(1e-3,peak+1)

number_func = 4
color = cm.rainbow(np.linspace(0, 1, number_func))

# trick_vec
file = 'cycles_trick_vector_naive.dat'
dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = np.copy(performance)
c = color[0]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c,label = 'trick_naive')
plt.xlim(dim[0], dim[-1])
	
ax.fill_between(dim, cum_perf, np.zeros(cum_perf.size), where=cum_perf >= 0, facecolor=c, interpolate=True)


# kernel
file = 'cycles_qmm_kernel_naive_trick.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[1]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c,label = 'kernel trick')
plt.xlim(dim[0], dim[-1])

ax.fill_between(dim, cum_perf, cum_perf-performance, where=cum_perf >= cum_perf-performance, facecolor=c, interpolate=True)

# add_trick_vector
file = 'cycles_add_vector_naive.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[2]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c,label = 'add_naive' )
plt.xlim(dim[0], dim[-1])

ax.fill_between(dim, cum_perf, cum_perf-performance, where=cum_perf >= cum_perf-performance, facecolor=c, interpolate=True)

# round/sat
file = 'cycles_round_saturation_naive.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[3]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c, label = 'round/sat')
plt.xlim(dim[0], dim[-1])

ax.fill_between(dim, cum_perf, cum_perf-performance, where=cum_perf >= cum_perf-performance, facecolor=c, interpolate=True)
plt.legend(loc = 'best')
plt.show()

fig.savefig(os.path.join(plot_folder, 'Cycles_trick.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Cycles_trick.png'), format='png',
            dpi=200)



############## QMM TRICK BLOCKING ##############################################

# Qmm_naive_trick_blocking = trick_vector_naive +
#                            qmm_kernel_trick_blocking +
#                            add_trick_vector_naive +
#                            round_saturation_naive

fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
ax.set_ylabel('[cycles]', rotation=0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Runtime of qmm with trick blocking VS matrix dimension")
# plt.ylim(1e-3,peak+1)

number_func = 4
color = cm.rainbow(np.linspace(0, 1, number_func))

# trick_vec
file = 'cycles_trick_vector_naive.dat'
dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = np.copy(performance)
c = color[0]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1.1*cum_perf[-1], 'trick_naive', color=c)


# kernel
file = 'cycles_qmm_kernel_trick_blocking.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[1]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1.1*cum_perf[-1], 'kernel blocking', color=c)


# add_trick_vector
file = 'cycles_add_vector_naive.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[2]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1.1*cum_perf[-1], 'add vector', color=c)


# round/sat
file = 'cycles_round_saturation_naive.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[3]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1.1*cum_perf[-1], 'round/sat', color=c)

plt.show()

fig.savefig(os.path.join(plot_folder, 'Cycles_blocking.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Cycles_blocking.png'), format='png',
            dpi=200)



############## QMM TRICK AVX ##############################################

# Qmm_naive_trick_AVX = trick_vector_AVX +
#                       qmm_kernel_blocking +
#                       add_trick_vector_AVX +
#                       round_saturation_avx

fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
ax.set_ylabel('[cycles]', rotation=0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Runtime of qmm with trick AVX VS matrix dimension")
# plt.ylim(1e-3,peak+1)

number_func = 4
color = cm.rainbow(np.linspace(0, 1, number_func))

# trick_vec
file = 'cycles_trick_vector_AVX.dat'
dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = np.copy(performance)
c = color[0]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])
ax.fill_between(dim, cum_perf, cum_perf-performance, where=cum_perf >= cum_perf-performance, facecolor=c, interpolate=True)

ax.text(dim[-3], 1.1*cum_perf[-1], 'trick_AVX', color=c)


# kernel
file = 'cycles_qmm_kernel_trick_blocking.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[1]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])
ax.fill_between(dim, cum_perf, cum_perf-performance, where=cum_perf >= cum_perf-performance, facecolor=c, interpolate=True)

ax.text(dim[-3], 1.1*cum_perf[-1], 'kernel AVX', color=c)


# add_trick_vector
file = 'cycles_add_vector_AVX.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[2]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])
ax.fill_between(dim, cum_perf, cum_perf-performance, where=cum_perf >= cum_perf-performance, facecolor=c, interpolate=True)

ax.text(dim[-3], 1.1*cum_perf[-1], 'add vector AVX', color=c)


# round/sat
file = 'cycles_round_saturation_AVX.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
cum_perf = cum_perf + performance
c = color[3]
plt.plot(dim, cum_perf, c=c, marker='d')
plt.plot(dim, cum_perf, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1.1*cum_perf[-1], 'round/sat', color=c)
ax.fill_between(dim, cum_perf, cum_perf-performance, where=cum_perf >= cum_perf-performance, facecolor=c, interpolate=True)

plt.show()

fig.savefig(os.path.join(plot_folder, 'Cycles_trick_AVX.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Cycles_trick_AVX.png'), format='png',
            dpi=200)



################## OVERALL COMPARISON ###############################
# Compare the runtime of the 5 different qmm

fig = plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
ax.set_ylabel('[cycles]', rotation=0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Runtime of qmm implementations VS matrix dimension")
# plt.ylim(1e-3,peak+1)

number_func = 5
color = cm.rainbow(np.linspace(0, 1, number_func))

# Naive
file = 'cycles_qmm_naive.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
c = color[0]
plt.plot(dim, performance, c=c, marker='d')
plt.plot(dim, performance, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], performance[-1], 'naive', color=c)


# trick
file = 'cycles_qmm_naive_trick.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
c = color[1]
plt.plot(dim, performance, c=c, marker='d')
plt.plot(dim, performance, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 0.8*performance[-1], 'naive trick', color=c)

# trick blocking
file = 'cycles_qmm_trick_blocking.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
c = color[3]
plt.plot(dim, performance, c=c, marker='d')
plt.plot(dim, performance, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 1*performance[-1], 'trick blocking', color=c)

# trick blocking
file = 'cycles_qmm_trick_blocking_AVX.dat'

dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
c = color[4]
plt.plot(dim, performance, c=c, marker='d')
plt.plot(dim, performance, c=c)
plt.xlim(dim[0], dim[-1])

ax.text(dim[-3], 0.6*performance[-1], 'trick blocking AVX', color=c)


plt.show()

fig.savefig(os.path.join(plot_folder, 'Cycles_qmm_comparison.eps'),
            format='eps')
fig.savefig(os.path.join(plot_folder, 'Cycles_qmm_comparison.png'), format='png',
            dpi=200)
	
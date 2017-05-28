import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm


prefix = 'Cycle_qmm'

files_qmm = [i for i in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder,i)) and prefix in i]

fig=plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of qmm VS matrix dimension")
plt.ylim(1e-3,peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, peak*np.ones(dim.shape),'k')
ax.text(dim[-5],7/10*peak,'scalar peak performance',color='k')
plt.show()

fig.savefig(os.path.join(plot_folder, 'Performance_qmm.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Performance_qmm.png'), format='png', dpi=200)



prefix = 'perf_add_vector'

files_qmm = [i for i in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder,i)) and prefix in i]

fig=plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of add_vector VS matrix dimension")
plt.ylim(1e-3,peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, peak*np.ones(dim.shape),'k')
ax.text(dim[-5],7/10*peak,'scalar peak performance',color='k')
plt.show()

fig.savefig(os.path.join(plot_folder, 'Performance_add_vector.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Performance_add_vector.png'), format='png', dpi=200)



prefix = 'perf_trick_vector'

files_qmm = [i for i in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder,i)) and prefix in i]

fig=plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of trick_vector VS matrix dimension")
plt.ylim(1e-3,peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, peak*np.ones(dim.shape),'k')
ax.text(dim[-5],7/10*peak,'scalar peak performance',color='k')
plt.show()

fig.savefig(os.path.join(plot_folder, 'Performance_trick_vector.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Performance_trick_vector.png'), format='png', dpi=200)



prefix = 'perf_round_saturation'

files_qmm = [i for i in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder,i)) and prefix in i]

fig=plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of trick_vector VS matrix dimension")
plt.ylim(1e-3,peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, peak*np.ones(dim.shape),'k')
ax.text(dim[-5],7/10*peak,'scalar peak performance',color='k')
plt.show()

fig.savefig(os.path.join(plot_folder, 'Performance_round_saturation.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Performance_round_saturation.png'), format='png', dpi=200)
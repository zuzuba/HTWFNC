import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm


data_folder = './data'
plot_folder = './plots'
scalar_int_peak = 8
scalar_flop_peak = 4
vector_int_peak = 32 
vector_flop_peak = 16
prefix = 'perf_quantize'
suffix='.dat'

files_quantize = [i for i in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder,i)) and prefix in i]

fig=plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of quantize VS matrix dimension")
plt.ylim(0,vector_flop_peak+1)

number_func = len(files_quantize)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_quantize,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)
	
plt.plot(dim, scalar_flop_peak*np.ones(dim.shape),'k')
plt.plot(dim, vector_flop_peak*np.ones(dim.shape),'k')
ax.text(dim[-5],11/10*scalar_flop_peak,'scalar flop peak performance',color='k')
ax.text(dim[-5],9.5/10*vector_flop_peak,'vector flop peak performance',color='k')
plt.show()
fig.savefig(os.path.join(plot_folder, 'Performance_quantize.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Performance_quantize.png'), format='png', dpi=200)



prefix = 'perf_qmm_kernel_'

files_qmm = [i for i in os.listdir(data_folder) if os.path.isfile(os.path.join(data_folder,i)) and prefix in i]

fig=plt.figure()
ax = plt.subplot(111)
plt.xlabel("n")
label = ax.set_ylabel('[flops/cycle]', rotation = 0)
ax.yaxis.set_label_coords(0, 1.05)
plt.grid()
plt.title("Performance of qmm kernel VS matrix dimension")
plt.ylim(0,scalar_int_peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, scalar_int_peak*np.ones(dim.shape),'k')
ax.text(dim[-5],9.5/10*scalar_int_peak,'scalar int peak performance',color='k')
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
plt.ylim(0,vector_int_peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, scalar_int_peak*np.ones(dim.shape),'k')
plt.plot(dim, vector_int_peak*np.ones(dim.shape),'k')
ax.text(dim[-5],9/10*scalar_int_peak,'scalar int peak performance',color='k')
ax.text(dim[-5],9.5/10*vector_int_peak,'vector int peak performance',color='k')
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
plt.ylim(0,vector_int_peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, scalar_int_peak*np.ones(dim.shape),'k')
plt.plot(dim, vector_int_peak*np.ones(dim.shape),'k')
ax.text(dim[-5],9/10*scalar_int_peak,'scalar int peak performance',color='k')
ax.text(dim[-5],9.5/10*vector_int_peak,'vector int peak performance',color='k')
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
plt.title("Performance of round/saturate VS matrix dimension")
plt.ylim(0,vector_flop_peak+1)

number_func = len(files_qmm)
color=cm.rainbow(np.linspace(0,1,number_func))

for file,c in zip(files_qmm,color):
	dim, performance = np.loadtxt(os.path.join(data_folder, file), unpack=True)
	plt.plot(dim, performance,c=c,marker='d')
	plt.plot(dim, performance,c=c)
	plt.xlim(dim[0], dim[-1])
	ax.text(dim[-3],11/10*performance[-1],file[len(prefix):-len(suffix)],color=c)

plt.plot(dim, scalar_flop_peak*np.ones(dim.shape),'k')
plt.plot(dim, vector_flop_peak*np.ones(dim.shape),'k')
ax.text(dim[-5],9/10*scalar_flop_peak,'scalar flop peak performance',color='k')
ax.text(dim[-5],9.5/10*vector_flop_peak,'vector flop peak performance',color='k')
plt.show()

fig.savefig(os.path.join(plot_folder, 'Performance_round_saturation.eps'), format='eps')
fig.savefig(os.path.join(plot_folder, 'Performance_round_saturation.png'), format='png', dpi=200)
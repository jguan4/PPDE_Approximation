from NN_Driver import NN_Driver
import numpy as np
import os, sys

arr = sys.argv
layers_str = arr[1]
layers = np.fromstring(layers_str[1:-1], dtype=int, sep=',')
env_toggle = arr[2]
sampling_method = int(arr[3])
Ns_str = arr[4]
Ntr = np.fromstring(Ns_str[1:-1], dtype=int, sep=',')
test_size = int(arr[5])
space_n = int(arr[6])
regular_alphas_str = arr[7]
regular_alphas = np.fromstring(regular_alphas_str[1:-1], dtype=float, sep=',')
train_toggle = int(arr[8])
type_weighting_str = arr[9]
type_weighting = np.fromstring(type_weighting_str[1:-1], dtype=float, sep=',')
net_toggle = arr[10]
opt_toggle = arr[11]
if len(arr)>12:
	L = int(arr[12])
else:
	L = 0
h = 1/space_n

kwargs = {"layers": layers, "env_toggle":env_toggle, "net_toggle":net_toggle, "opt_toggle":opt_toggle, "sampling_method":sampling_method, "Ntr":Ntr, "test_size":test_size, "h":h, "regular_alphas":regular_alphas, "type_weighting":type_weighting, "L":L}
pinn = NN_Driver(**kwargs)

if sampling_method == 0:
	method_str = "PODNN"
else:
	method_str = "PINN"
PATH_L = "./Log/Logs/{1}/{0}/".format(env_toggle,method_str)
folder_path = PATH_L + "Net{6}_Layers{0}_Ntr{1}_h{2}_Reg{3}_Sample{4}_Weight{5}_Opt{7}/".format(layers, Ntr, int(1/h), regular_alphas, sampling_method, type_weighting, net_toggle, opt_toggle)
record_path = PATH_L + "rel_errs.csv"

if train_toggle == 0:
	save_toggle = False
	pinn.train_nn(save_toggle)
	pinn.test_nn()
elif train_toggle == 1:
	pinn.train_nn()
	pinn.test_nn()
elif train_toggle == 2:
	pinn.load_nn(None)
	pinn.train_nn()
	pinn.test_nn()	
elif train_toggle == 3:
	pinn.load_nn(None)
	figure_path = folder_path+"Data/"
	pinn.plot_test(figure_path)
elif train_toggle == 4:
	pinn.load_nn(None)
	pinn.test_nn(record_path = record_path)

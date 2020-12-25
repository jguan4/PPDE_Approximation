import numpy as np
from CD_1d import CD_1D
from NN_tf import NN_tf
from train_lm import train_lm
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import time
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# setup directory to save weights in
PATH_W = "./Weights/CD_1D/"
if not os.path.exists(PATH_W):
	os.makedirs(PATH_W)
# parameters to play with
h = 1/2048 # mesh size for creating testing solutions
Nf = 1000 # number of residual samples
Nd = 100 # number of Dirichlet boundary samples
N0 = 0 # number of true solutions
type_weighting = [1,1,1] # weighting for each type of samples in the loss function
layers = [32,32,32] # hidden layer sizes 
N_test = 10 # number of testing xi's

# setup the problem and network
Ntr = [Nf,Nd,N0]
env = CD_1D(h)
net = NN_tf(env.state_space_size, env.output_space_size, layers, env)
samples_list = env.generate_PINN_samples(Nf, Nd, N0, type_weighting)
path_weight = PATH_W + "Layers{0}_Ntr{1}_h{2}_Weight{3}.npz".format(layers, Ntr, int(1/h), type_weighting)

train = False # if True, we will start training and test; if False, we will load previously trained weights and test

if train:
	# parameters for training
	Epoch_max = 100
	tol = 1e-7
	mu = 1e-2
	beta = 2
	# starts training
	train_lm(net, samples_list, Epoch_max, tol, mu, beta, path_weight = path_weight)

	# test and plot
	env.plot_NN(net, N_test)
else:
	# load weights
	net.load_weights_biases(path_weight)
	# test and plot
	env.plot_NN(net, N_test)


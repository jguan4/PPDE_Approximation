import numpy as np
from Environments.Poisson_1D import Poisson_1D
from Environments.CD_comp_steady import CD_comp_steady
from Environments.CD_1D import CD_1D
from Environments.CD_2D_1 import CD_2D_1
from Environments.CD_2D_2 import CD_2D_2
from Environments.CD_2D_3 import CD_2D_3
from Environments.CD_2D_4 import CD_2D_4
from Environments.CD_2D_5 import CD_2D_5
from Environments.CD_2D_6 import CD_2D_6
from Environments.CD_2D_7 import CD_2D_7
from Environments.CD_1D_1 import CD_1D_1
from Environments.CD_1D_2 import CD_1D_2
from Environments.CD_1D_3 import CD_1D_3
from Environments.CD_1D_4 import CD_1D_4
from Environments.CD_1D_5 import CD_1D_5
from Environments.CD_1D_6 import CD_1D_6
from Environments.CD_1D_7 import CD_1D_7
from Environments.CD_1D_8 import CD_1D_8
from Environments.CD_1D_9 import CD_1D_9
from Environments.CD_1D_10 import CD_1D_10
from Environments.CD_1D_11 import CD_1D_11
from Environments.CD_1D_12 import CD_1D_12
from Environments.CD_1D_13 import CD_1D_13
from Environments.CD_1D_14 import CD_1D_14
from Environments.CD_1D_15 import CD_1D_15
from Environments.CD_1D_16 import CD_1D_16
from Environments.CD_1D_17 import CD_1D_17
from Environments.Burgers_Equation import Burgers_Equation
from NN.NN_tf import NN_tf
from NN.ResNet_tf import ResNet_tf
from NN.RNN_tf import RNN_tf
from Optimizers.train_lm import train_lm
from Optimizers.train_Adam import train_Adam
from Optimizers.train_sgd import train_sgd
from Optimizers.train_lbfgs import train_lbfgs
from Optimizers.train_Newton import train_Newton
from Visualization.loss_plot import Loss_Vis
from Visualization.res_plot import Res_Vis
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import time
import scipy.io
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

PATH = "./Log/"

class NN_Driver:
	def __init__(self, layers, env_toggle, net_toggle, opt_toggle, sampling_method, Ntr, test_size, h, regular_alphas, type_weighting, app_str="", L=0):
		global PATH
		self.start_time = time.time()
		# save parameters
		self.layers = layers # network sizes
		self.env_toggle = env_toggle # environment type
		self.net_toggle = net_toggle # network type
		self.opt_toggle = opt_toggle # optimization type
		# sampling method 	0: POD-NN RB; 
		# 					1: PINN with residual loss on solution; 
		# 					2: PINN without residual loss on solution;
		# 					3: PINN using RNN
		self.sampling_method = sampling_method
		self.Ntr = Ntr # train sample sizes [Nf, Nb_d, Nb_n, N0]
		self.test_size = test_size # number of testing samples
		self.h = h # stepsize for testing solutions
		self.L = L
		self.regular_alphas = regular_alphas # regularization parameter
		# weighting on each part of the loss in PINN
		self.type_weighting = type_weighting 
		self.app_str = app_str

		if self.sampling_method == 0:
			self.method_str = "PODNN"
		else:
			self.method_str = "PINN"
		PATH_W = PATH + "Weights/{1}/{0}/".format(self.env_toggle,self.method_str)
		PATH_L = PATH + "Logs/{1}/{0}/".format(self.env_toggle,self.method_str)
		if not os.path.exists(PATH_W):
			os.makedirs(PATH_W)
		if not os.path.exists(PATH_L):
			os.makedirs(PATH_L)
		htemp = 1/2048
		self.save_name = "Net{6}_Layers{0}_Ntr{1}_h{2}_Reg{3}_Sample{4}_Weight{5}_Opt{7}_L{8}{9}".format(self.layers, Ntr, int(1/htemp), regular_alphas, self.sampling_method, type_weighting, self.net_toggle,self.opt_toggle, self.L, self.app_str)
		self.path_weight = PATH_W + self.save_name+ ".npz"
		if self.method_str == "PODNN":
			self.path_log = PATH_L + "Net{6}_Layers{0}_Ntr{1}_h{2}_Reg{3}_Sample{4}_Weight{5}_Opt{7}/L{8}/".format(self.layers, Ntr, int(1/htemp), regular_alphas, self.sampling_method, type_weighting, self.net_toggle,self.opt_toggle,self.L)
		elif self.method_str == "PINN":
			self.path_log = PATH_L + "{0}/".format(self.save_name)

		if not os.path.exists(self.path_log):
			os.makedirs(self.path_log)
		self.path_env = "./Environments/{0}/".format(self.env_toggle)
		if not os.path.exists(self.path_env):
			os.makedirs(self.path_env)

		self.Ntr_name = ["Res", "B_D", "B_N", "Init"]
		self.Ntr_t = np.any(np.array([Ntr]),axis=0)
		
		self.initialize_env()
		self.input_size = self.env.state_space_size
		self.output_size = self.env.output_space_size
		if self.sampling_method == 0:
			self.output_size = self.L
		self.initialize_net()
		self.process_training_samples()

	def initialize_env(self):
		if self.env_toggle == "Poisson":
			self.env = Poisson_1D(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_envnn = self.path_env, L = self.L)
		elif self.env_toggle == "CD_steady":
			self.env = CD_comp_steady(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "Burger":
			self.env = Burgers_Equation(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D":
			self.env = CD_1D(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_1":
			self.env = CD_1D_1(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_2":
			self.env = CD_1D_2(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_3":
			self.env = CD_1D_3(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_4":
			self.env = CD_1D_4(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_5":
			self.env = CD_1D_5(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_6":
			self.env = CD_1D_6(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_7":
			self.env = CD_1D_7(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_8":
			self.env = CD_1D_8(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_9":
			self.env = CD_1D_9(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_10":
			self.env = CD_1D_10(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_11":
			self.env = CD_1D_11(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_12":
			self.env = CD_1D_12(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_13":
			self.env = CD_1D_13(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_14":
			self.env = CD_1D_14(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_15":
			self.env = CD_1D_15(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_16":
			self.env = CD_1D_16(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_1D_17":
			self.env = CD_1D_17(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_2D_1":
			self.env = CD_2D_1(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_2D_2":
			self.env = CD_2D_2(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_2D_3":
			self.env = CD_2D_3(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_2D_4":
			self.env = CD_2D_4(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_2D_5":
			self.env = CD_2D_5(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_2D_6":
			self.env = CD_2D_6(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)
		elif self.env_toggle == "CD_2D_7":
			self.env = CD_2D_7(self.Ntr, self.test_size, self.h, type_weighting = self.type_weighting, sampling_method = self.sampling_method, path_env = self.path_env, L = self.L)

	def initialize_net(self):
		if self.net_toggle == "Dense":
			self.net = NN_tf(self.input_size, self.output_size, self.layers, self.env, self.regular_alphas)
		elif self.net_toggle == "Res":
			self.net = ResNet_tf(self.input_size, self.output_size, self.layers, self.env, self.regular_alphas)
		elif self.net_toggle == "RNN":
			self.net = RNN_tf(self.input_size, self.output_size, self.layers, self.env, self.regular_alphas)

	def process_training_samples(self):
		if self.net_toggle != "RNN":
			if self.sampling_method == 0:
				self.samples_list = self.env.generate_POD_samples(self.app_str)
			elif self.sampling_method == 1 or self.sampling_method == 2:
				self.samples_list = self.env.generate_PINN_samples(self.app_str)
		else:
			self.sampling_method = 3
			self.samples_list = self.env.generate_RNN_samples()

	def train_nn(self, save_for_plot = False, save_toggle = True):
		if self.opt_toggle == "lm":
			# third input: maximum epoch; fourth input: tolerance; fifth: starting lambda parameter; sixth: factor to change lambda
			if save_for_plot:
				self.plot_weight_foldername = "./Plot/Data/Weights/{1}/{0}/{2}".format(self.env_toggle,self.method_str,self.save_name)
				if not os.path.exists(self.plot_weight_foldername):
					os.makedirs(self.plot_weight_foldername)
					train_lm(self.net, self.samples_list, 1000, 1e-7, 1e-2, 2, save_toggle, save_for_plot, path_weight = self.path_weight, path_log = self.path_log, path_plot = self.plot_weight_foldername)
				else:
					pass
			else:
				train_lm(self.net, self.samples_list, 1000, 1e-7, 1e-2, 2, save_toggle, save_for_plot, path_weight = self.path_weight, path_log = self.path_log)

		elif self.opt_toggle == "adam":
			train_Adam(self.net, self.samples_list, 30000, 1e-7, save_toggle, save_for_plot, path_weight = self.path_weight, path_log = self.path_log)
		elif self.opt_toggle == "lbfgs":
			train_lbfgs(self.net, self.Xs_list, self.us_list, 1000, 1e-9, save_toggle, m=100, path_weight = self.path_weight, path_log = self.path_log)
		elif self.opt_toggle == "sgd":
			train_sgd(self.net, self.Xs_list, self.us_list, 1000, 1e-9, save_toggle, path_weight = self.path_weight, path_log = self.path_log)
		print("Time for training: {0}".format(time.time()-self.start_time))

	def test_nn(self, record_path = None):
		self.env.test_NN(self.net, None, self.app_str)
		print("Time up to testing: {0}".format(time.time()-self.start_time))

	def load_nn(self, path):
		print(self.env.name,self.save_name)
		if path is None:
			if not os.path.exists(self.path_weight):
				check_path_weight = self.path_weight.replace('_lm','')
				if os.path.exists(check_path_weight):
					self.net.load_weights_biases(check_path_weight)
			else:
				self.net.load_weights_biases(self.path_weight)
		else:
			self.net.load_weights_biases(path)

	def plot_test(self, figure_path = None):
		self.env.plot_NN(self.net, figure_path)

	def save_test(self, data_path = None):
		if data_path is not None:
			if not os.path.exists(data_path):
			    os.makedirs(data_path)
		data_name = "{1}_{0}.mat".format(self.save_name,self.env.name)
		u_test_grid, u_test_p_grid, _, _, _ = self.env.test_NN(self.net, data_path, data_name)
		if data_path is not None:
			scipy.io.savemat(data_path+"/"+data_name, {'true_solution':u_test_grid.numpy(), 'approximation': u_test_p_grid.numpy(), 'xi':self.env.mu_mat_test, 'x':self.env.x})

	def visualize_loss(self):
		lv = Loss_Vis(self.net,self.samples_list)
		lv.visualize_space()

	def visualize_path(self):
		lv = Loss_Vis(self.net,self.samples_list)
		lv.visualize_path(self.plot_weight_foldername)

	def visualize_path_lev(self):
		lv = Loss_Vis(self.net,self.samples_list)
		lv.visualize_path_lev(self.plot_weight_foldername)

	def visualize_residual(self):
		rv = Res_Vis(self.net,self.samples_list,self.save_name,self.env.name)
		rv.generate_res_mat(self.plot_weight_foldername)
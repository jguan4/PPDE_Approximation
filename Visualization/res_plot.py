import numpy as np
import sys
import os 
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import scipy.io

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
tf.random.set_seed(5)

class Res_Vis:
	def __init__(self, net, samplelist, savename,env_name):
		self.net = net
		self.samplelist = samplelist
		# self.create_uniform_mesh()
		self.savepath = "./Plot/Data/PINN/"+env_name+"/"+savename+"_residual"
		
	def generate_res_mat(self,folder):
		self.stored_weights_path = folder
		if not os.path.exists(self.savepath):
		    os.makedirs(self.savepath)
		list_of_files  = os.listdir(self.stored_weights_path)
		paths = [os.path.join(self.stored_weights_path, basename) for basename in list_of_files]
		latest_file = max(paths, key=os.path.getctime)
		last_weight_name = os.path.basename(latest_file)
		self.last_iter = int(last_weight_name.split('.')[0])
		
		npzfile = np.load(latest_file)
		final_weight = npzfile['weights_lin']
		final_biase = npzfile['biases_lin']

		weight_diff = np.zeros([np.prod(final_weight.shape),self.last_iter])
		biases_diff = np.zeros([np.prod(final_biase.shape),self.last_iter])

		# x = np.linspace(self.net.env.domain[0,0],self.net.env.domain[0,1],num=self.net.env.N)
		# y = np.linspace(self.net.env.domain[1,0],self.net.env.domain[1,1],num=self.net.env.N)
		# X, Y = np.meshgrid(x,y)
		# X_vec = X.reshape([self.net.env.N**2,1])
		# Y_vec = Y.reshape([self.net.env.N**2,1])
		# xi_vec = np.ones([self.net.env.N**2,1])*1e-4
		# target = np.zeros([self.net.env.N**2,1])
		# t_tf = tf.constant((),shape = (self.net.env.N**2,0),dtype = tf.float32)
		# x_tf = tf.constant(X_vec,dtype = tf.float32)
		# y_tf = tf.constant(Y_vec,dtype = tf.float32)
		# xi_tf = tf.constant(xi_vec,dtype = tf.float32)
		# target_tf = tf.constant(target, dtype = tf.float32)

		for epoch in range(self.last_iter):
			filename = self.stored_weights_path+"/{0}.npz".format(epoch)
			npzfile = np.load(filename)
			weights_lin = npzfile['weights_lin']
			biases_lin = npzfile['biases_lin']
			loss_val = self.net.set_weights_loss(self.samplelist, weights_lin, biases_lin)

			for i in range(len(self.samplelist)):
				dict_i = self.samplelist[i]
				name_i = dict_i["type"]
				x_tf = dict_i["x_tf"]
				y_tf = dict_i["y_tf"]
				t_tf = dict_i["t_tf"]
				xi_tf = dict_i["xi_tf"]
				target = dict_i["target"]
				N = dict_i["N"]
				weight = dict_i["weight"]


				if name_i == "Res":
					f_res = self.net.compute_residual(x_tf, y_tf, t_tf, xi_tf, target)
					ur, u_xr, u_yr, u_tr, u_xxr, u_yyr, u_xyr = self.net.derivatives(x_tf, y_tf, t_tf, xi_tf)
					
					data_name = "res_{0}.mat".format(epoch)
					scipy.io.savemat(self.savepath+"/"+data_name, {"res_x":x_tf.numpy(),"res_y":y_tf.numpy(),"res_xi":xi_tf.numpy(),"res":f_res.numpy(),"u":ur.numpy(),"u_x":u_xr.numpy(),"u_y":u_yr.numpy(),"u_xx":u_xxr.numpy(),"u_yy":u_yyr.numpy()})

				elif name_i == "B_D":
					err_do = self.net.compute_solution(x_tf, y_tf, t_tf, xi_tf, target)
					ubd, u_xbd, u_ybd, u_tbd, u_xxbd, u_yybd, u_xybd = self.net.derivatives(x_tf, y_tf, t_tf, xi_tf)
						
				elif name_i == "B_N":
					err_n = self.net.compute_neumann(x_tf, y_tf, t_tf, xi_tf,target)
					ubn, u_xbn, u_ybn, u_tbn, u_xxbn, u_yybn, u_xybn = self.net.derivatives(x_tf, y_tf, t_tf, xi_tf)

	def create_uniform_mesh(self,p=1e-4):
		self.samplelist = []
		x = np.linspace(self.net.env.domain[0,0],self.net.env.domain[0,1],num=self.net.env.N)
		y = np.linspace(self.net.env.domain[1,0],self.net.env.domain[1,1],num=self.net.env.N)
		X, Y = np.meshgrid(x,y)
		X_vec = X.reshape([self.net.env.N**2,1])
		Y_vec = Y.reshape([self.net.env.N**2,1])
		xi_vec = np.ones([self.net.env.N**2,1])*p
		target = np.zeros([self.net.env.N**2,1])
		t_tf = tf.constant((),shape = (self.net.env.N**2,0),dtype = tf.float32)
		x_tf = tf.constant(X_vec,dtype = tf.float32)
		y_tf = tf.constant(Y_vec,dtype = tf.float32)
		xi_tf = tf.constant(xi_vec,dtype = tf.float32)
		target_tf = tf.constant(target, dtype = tf.float32)
		N = tf.constant(self.net.env.N**2, dtype = tf.float32)
		weight = tf.constant(1, dtype = tf.float32)
		sample_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Res', 'weight':weight}
		self.samplelist.append(sample_dict)

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from smt.sampling_methods import LHS
import scipy.io
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import csv

class CD_1D_7:
	def __init__(self, N_p_train, N_p_test, h, type_weighting = [1,1,1,1], inner = False, sampling_method = 0, add_sample = False, path_env = "./Environments/", L = 0):
		self.name = "CD_1D_7"
		self.sampling_method = sampling_method
		self.u_dim = 1
		self.P_dim = 1
		self.domain = np.array([[0,1]])
		self.plimits = np.array([[-4,0]])
		self.x_p_domain = np.array([[0, 1], [-4, 0]])
		self.h = h
		self.path_env = path_env
		self.type_weighting = type_weighting
		self.L = L

		# return full solutions for parameters generated
		if self.sampling_method == 0:
			self.lb = np.array([1e-4])
			self.ub = np.array([1.0])
			self.state_space_size = self.P_dim
			self.output_space_size = None
			self.N_p_train = N_p_train[3]
			self.P_samples = np.array([])
			self.u_samples = np.array([])
			self.u_tests = np.array([])
			self.var_list = [[0,1]]
			self.generate_para()

		# return point-wise solutions, including parameters in input
		elif self.sampling_method == 1 or self.sampling_method == 2:
			self.state_space_size = self.u_dim+self.P_dim
			self.output_space_size  = 1
			self.lb = np.array([0.0, 1e-4])
			self.ub = np.array([1e4, 1.0])
			self.Nf = N_p_train[0]
			self.Nb = N_p_train[1]
			self.Nn = N_p_train[2]
			self.N0 = N_p_train[3]

			h_sample = 2/np.ceil(np.sqrt(np.max(N_p_train)))

			if h_sample<h:
				h = h_sample
			self.N_p_train = int(self.N0**(1/self.state_space_size))

			self.N0_samples = np.array([])
			self.u0_samples = np.array([])

			self.N0_tests = np.array([])
			self.u0_tests = np.array([])

		elif self.sampling_method == 3:
			self.plimits = np.array([[-1,0]])
			self.x_p_domain = np.array([[0, 1], [-1, 0]])
			self.lb = np.array([1e-1])
			self.ub = np.array([1.0])
			self.state_space_size = self.P_dim
			self.output_space_size  = 1
			self.N_p_train = N_p_train[3]
			N = int(1/h)+1
			self.x_disc = np.linspace(self.domain[0,0],self.domain[0,1],num=N)
			self.x_disc = self.x_disc[1::]
			self.x_dim = 1
			self.p_dim = 1
			self.generate_para()
			self.generate_RNN_init()

		self.N_p_test = N_p_test

		self.N = int(1/h)+1
		self.nx = self.N-2
		self.inner = inner
		self.x = np.linspace(self.domain[0,0],self.domain[0,1],num=self.N)
		self.x_in = self.x[1:-1]


	def generate_para(self):
		np.random.seed(10)
		sampling = LHS(xlimits=self.plimits)
		filename = self.path_env+"CD_1D_{0}.npy".format(self.N_p_train)
		
		# check if train parameters exist
		if os.path.exists(filename):
			self.mu_mat_train = np.load(filename)
		else: 
			self.mu_mat_train = sampling(self.N_p_train).T
			self.mu_mat_train[0,:] = np.power(10,self.mu_mat_train[0,:])
			np.save(filename,self.mu_mat_train)

	def generate_para_test(self):
		np.random.seed(10)
		sampling = LHS(xlimits=self.plimits)
		# check if test parameters exist
		if os.path.exists(self.path_env+"CD_1D_{0}.npy".format(self.N_p_test)):
			self.mu_mat_test = np.load(self.path_env+"CD_1D_{0}.npy".format(self.N_p_test))
		else:
			self.mu_mat_test = sampling(self.N_p_test).T
			self.mu_mat_test[0,:] = np.power(10,self.mu_mat_test[0,:])
			np.save(self.path_env+"CD_1D_{0}.npy".format(self.N_p_test),self.mu_mat_test)

	def u_exact_train(self):
		if self.sampling_method == 0:
			for i in range(self.mu_mat_train.shape[1]):
				p = self.mu_mat_train[:,i]
				self.generate_one_sol(p)
			return [self.mu_mat_train.T, self.u_samples]
		elif self.sampling_method == 3:
			return self.generate_RNN_samples()

	def u_exact_test(self):
		self.generate_para_test()
		if self.sampling_method == 0:
			self.u_tests = np.array([])
			for i in range(self.mu_mat_test.shape[1]):
				p = self.mu_mat_test[:,i]
				self.generate_one_sol(p, test = True)			
			return self.generate_POD_tests()
		elif self.sampling_method == 1 or self.sampling_method == 2:
			self.N0_tests = np.array([])
			self.u0_tests = np.array([])
			for i in range(self.mu_mat_test.shape[1]):
				p = self.mu_mat_test[:,i]
				self.generate_one_sol(p, test = True)
			return self.generate_PINN_tests()
		elif self.sampling_method == 3:
			self.u_tests = np.array([])
			for i in range(self.mu_mat_test.shape[1]):
				p = self.mu_mat_test[:,i]
				self.generate_one_sol(p, test = True)
			return self.generate_RNN_tests()

	def u_exact(self, X):
		x = X[:,[0]]
		xi = X[:,[1]]
		u = 1-xi*x-xi*(np.exp(-x)-np.exp(-1/xi))
		# u = 1-np.exp((x-1)/xi)
		return u

	def generate_A_F(self,p):
		xi = p[0]
		reynolds_num = self.h/(2*xi)
		L = sparse.diags([-(1+reynolds_num),2,-(1-reynolds_num)],[-1,0,1],shape=(self.nx,self.nx), format = 'csr')
		F = np.zeros((self.nx,1))
		F[0] = (1+reynolds_num)*(1-np.exp(-1/xi))
		return L,F

	def generate_one_sol(self, p, test = False):
		L,F = self.generate_A_F(p)
		u = spsolve(L,F)
		self.compile_output(u, p, test)

	def compile_output(self, u, p, test):
		if self.sampling_method == 0 or self.sampling_method == 3:
			if self.inner:
				U = u.reshape((self.nx,1))
			else:
				U = self.fill_BC(u, p)
			if not test:		
				self.u_samples = np.hstack((self.u_samples,U)) if self.u_samples.size else U
			else:
				xi = p[0]*np.ones([self.N,1])
				x = self.x.reshape([self.N,1])
				X = np.concatenate((x,xi),axis=1)
				U = self.u_exact(X)
				self.u_tests = np.hstack((self.u_tests,U)) if self.u_tests.size else U
		elif self.sampling_method == 1 or self.sampling_method == 2:
			if not test:
				u_inner = u.reshape((self.nx,1))
				X_f = self.create_inner_X(p)

				self.N0_samples = np.concatenate((self.N0_samples, X_f),axis = 0) if self.N0_samples.size else X_f
				self.u0_samples = np.concatenate((self.u0_samples,u_inner),axis = 0) if self.u0_samples.size else u_inner
			else:
				X_0 = self.create_tol_X(p)
				u_tol = self.u_exact(X_0)
				self.N0_tests = np.concatenate((self.N0_tests, X_0),axis = 0) if self.N0_tests.size else X_0
				self.u0_tests = np.concatenate((self.u0_tests, u_tol),axis = 0) if self.u0_tests.size else u_tol				
				
	def fill_BC(self,inner_u,p):
		xi = p[0]
		U = np.zeros((self.N,1))
		U[1:-1,0] = inner_u
		U[0] = (1-np.exp(-1/xi))
		return U

	def create_inner_X(self,p):
		X_in = self.x_in.reshape((self.N-2,1))
		P = p*np.ones((self.nx,self.P_dim))
		X_f = np.concatenate((X_in,P),axis=1)
		return X_f

	def create_tol_X(self,p):
		X = self.x.reshape((self.N,1))
		X = (1-X)/p
		P = p*np.ones((self.N,self.P_dim))
		X_f = np.concatenate((X,P),axis=1)
		return X_f

	def generate_RNN_init(self):
		self.h_init = 1-np.exp(-1/self.mu_mat_train)
		self.h_init = np.transpose(self.h_init)

	def generate_RNN_samples(self):
		xi_tf = tf.constant(np.transpose(self.mu_mat_train),dtype = tf.float32)
		target_tf = tf.constant(1-np.exp(((self.x-1))/np.transpose(self.mu_mat_train)),dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		x_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		N = tf.constant(self.N_p_train, dtype = tf.float32)
		weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
		self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Init', 'weight':weight}
		samples_list = [self.X0_dict]
		return samples_list

	def generate_RNN_tests(self):
		xi_tf = tf.constant(self.mu_mat_test.T ,dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		x_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		N = tf.constant(self.N_p_test, dtype = tf.float32)
		weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'N':N, 'type':'Init', 'weight':weight}
		self.h_init = 1-np.exp(-1/self.mu_mat_test)
		self.h_init = np.transpose(self.h_init)
		return X0_dict, self.u_tests

	def generate_POD_samples(self, app_str):
		self.generate_para(app_str)
		p_train, u_train = self.u_exact_train()
		if os.path.exists(self.path_env+"{2}_{1}{3}_V_{0}.npy".format(self.L, self.N_p_train, self.name,app_str)):
			self.V = np.load(self.path_env+"{2}_{1}{3}_V_{0}.npy".format(self.L, self.N_p_train, self.name,app_str))
		else:
			u,s,v = np.linalg.svd(u_train) 
			self.V = u[:,0:self.L]
			np.save(self.path_env+"{2}_{1}{3}_V_{0}.npy".format(self.L, self.N_p_train, self.name,app_str),self.V)
		p_batch = p_train
		u_batch = u_train
		u_batch = self.V.T@u_batch
		xi_tf = tf.constant(p_batch,dtype = tf.float32)
		target_tf = tf.constant(u_batch.T,dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		x_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		N = tf.constant(self.N_p_train, dtype = tf.float32)
		weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
		self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Init', 'weight':weight}
		samples_list = [self.X0_dict]
		return samples_list
	
	def generate_POD_tests(self):
		xi_tf = tf.constant(self.mu_mat_test.T ,dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		x_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		N = tf.constant(self.N_p_test, dtype = tf.float32)
		weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'N':N, 'type':'Init', 'weight':weight}
		return X0_dict, self.u_tests

	def generate_PINN_samples(self, app_str = ""):

		Ns = [self.Nf, self.Nb, self.Nn, self.N0]
		samples_list = []

		filename = "CD_1D_7ver_{0}{1}.npz".format(Ns,app_str)
		if os.path.exists("{1}{0}".format(filename,self.path_env)):
			npzfile = np.load("{1}{0}".format(filename,self.path_env))
			if self.Nf>0:
				self.Xf = npzfile['Xf']
				target_f = np.zeros([self.Nf,1])
			else:
				self.Xf_tf = None
			if self.Nb>0:
				self.Xb_d = npzfile['Xb_d']
				self.ub_d = npzfile['ub_d']
			if self.Nn>0:
				self.Xb_n = npzfile['Xb_n']
				self.ub_n = npzfile['ub_n']
			if self.N0>0:
				self.X0 = npzfile['X0']
				self.u0 = npzfile['u0']
			else:
				self.X0_tf = None
				self.u0_tf = None
		else:
			np.random.seed(10)
			if app_str == "_uniform":
				xnum = 100
				epsnum = int(self.Nf/xnum)
				eps_log = np.linspace(-4,0,epsnum)
				eps = np.power(10,eps_log)
				eps_arr = np.tile(eps,xnum)
				eps_arr = eps_arr.reshape((self.Nf,1))
				rend = 1/eps
				xmat = np.linspace(np.zeros(rend.shape),rend,xnum+2)
				xmat = xmat[1:-1]
				x_arr = xmat.reshape((self.Nf,1))
				self.Xf = np.concatenate((x_arr,eps_arr),axis=1)
			elif app_str == "_uniform1":
				xnum = 100
				epsnum = int(self.Nf/xnum)
				# eps_log = np.linspace(-4,0,epsnum)
				eps = np.linspace(1e-4,1,epsnum)
				eps_arr = np.tile(eps,xnum)
				eps_arr = eps_arr.reshape((self.Nf,1))
				rend = 1/(eps)
				xmat = np.linspace(np.zeros(rend.shape),rend,xnum+2)
				xmat = xmat[1:-1]
				x_arr = xmat.reshape((self.Nf,1))
				self.Xf = np.concatenate((x_arr,eps_arr),axis=1)
			else:
				sampling_f = LHS(xlimits = self.x_p_domain)
				self.Xf = sampling_f(self.Nf)
				self.Xf[:,1] = np.power(10, self.Xf[:,1])
				self.Xf[:,0] = (1-self.Xf[:,0])/self.Xf[:,1]

			target_f = np.zeros([self.Nf,1])

			sampling_b = LHS(xlimits = np.array([[-4, 0]]))
			x_p_b = sampling_b(self.Nb)
			pb = x_p_b
			pb_10= np.power(10, pb)
			lb = np.concatenate((1/pb_10,pb_10),axis = 1)
			ulb = np.zeros((self.Nb,1))

			self.Xb_d = lb
			self.ub_d = ulb

			sampling_n = LHS(xlimits = np.array([[-4, 0]]))
			x_p_n = sampling_n(self.Nn)
			pn = x_p_n
			pn_10= np.power(10, pn)
			rb = np.concatenate((np.zeros((self.Nn,1)),pn_10),axis = 1)
			urb = np.zeros((self.Nn,1))

			self.Xb_n = rb
			self.ub_n = urb

			if self.N0>0:
				sampling_0 = LHS(xlimits = self.x_p_domain)
				x = sampling_0(self.N0)
				x[:,1] = np.power(10, x[:,1])
				x[:,0] = (1-x[:,0])/x[:,1]

				str_arr = app_str.split("_")
				if len(str_arr)==3:
					setting_str = str_arr[1]
					setting_para_str = str_arr[2]
					setting_para_str = setting_para_str.replace("p",".")
					setting = int(setting_str)
					setting_para = float(setting_para_str)
				else:
					setting = 0
				# setting 1
				if setting == 1:
					x[:,0] = x[:,1]*x[:,0]+1-x[:,1]
				# setting 2
				elif setting == 2:
					x[:,0] = setting_para*x[:,1]*x[:,0]+1-setting_para*x[:,1]
					x[:,0] = np.maximum(0.001*np.ones(np.shape(x[:,0])),x[:,0])
				# setting 3
				elif setting == 3:
					percent = setting_para/100
					in_corner_int = int(percent*self.N0)
					out_corner_int = self.N0-in_corner_int
					N0_sel = np.random.choice(self.N0,in_corner_int, replace=False)
					N0_sel_diff = np.array([i for i in range(self.N0) if (i not in N0_sel)])
					N0_sel = N0_sel.reshape([in_corner_int,1])
					N0_sel_diff = N0_sel_diff.reshape([out_corner_int,1])
					x[N0_sel,0] = x[N0_sel,1]*x[N0_sel,0]+1-x[N0_sel,1]
					x[N0_sel_diff,0] = (1-x[N0_sel_diff,1])*x[N0_sel_diff,0]

				# self.X0 = np.concatenate((x,1e-4*np.ones((self.N0,1))),axis = 1)
				self.X0 = x
				self.u0 = self.u_exact(self.X0)
				np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, Xb_n = self.Xb_n, ub_n = self.ub_n, X0 = self.X0, u0 = self.u0)
			else:
				np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, Xb_n = self.Xb_n, ub_n = self.ub_n)
		
		# fig, ax = plt.subplots()
		# ax.scatter(1-self.Xf[:,0]*self.Xf[:,1], self.Xf[:,1], color ="red")
		# ax.scatter(1-self.Xb_d[:,0]*self.Xb_d[:,1], self.Xb_d[:,1])
		# plt.show()

		if self.Nf>0:
			y_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
			t_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
			x_tf = tf.constant(self.Xf[:,[0]],dtype = tf.float32)
			xi_tf = tf.constant(self.Xf[:,[1]],dtype = tf.float32)
			target_tf = tf.constant(target_f, dtype = tf.float32)
			N = tf.constant(self.Nf, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[0], dtype = tf.float32)
			self.Xf_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Res', 'weight':weight}
			samples_list.append(self.Xf_dict)

		if self.Nb>0:
			y_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
			t_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
			x_tf = tf.constant(self.Xb_d[:,[0]],dtype = tf.float32)
			xi_tf = tf.constant(self.Xb_d[:,[1]],dtype = tf.float32)
			target_tf = tf.constant(self.ub_d, dtype = tf.float32)
			N = tf.constant(self.Nb, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[1], dtype = tf.float32)
			self.Xb_d_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_D', 'weight':weight}
			samples_list.append(self.Xb_d_dict)

		if self.Nn>0:
			y_tf = tf.constant((),shape = (self.Nn,0),dtype = tf.float32)
			t_tf = tf.constant((),shape = (self.Nn,0),dtype = tf.float32)
			x_tf = tf.constant(self.Xb_n[:,[0]],dtype = tf.float32)
			xi_tf = tf.constant(self.Xb_n[:,[1]],dtype = tf.float32)
			target_tf = tf.constant(self.ub_n, dtype = tf.float32)
			N = tf.constant(self.Nn, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[1], dtype = tf.float32)
			self.Xb_n_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_N', 'weight':weight}
			samples_list.append(self.Xb_n_dict)

		if self.N0>0:
			#check number of samples in (1-xi,1) corner
			xis = self.X0[:,[1]]
			xs = self.X0[:,[0]]
			inds_corner = [i for i in range(len(xs)) if xs[i]>=1-xis[i]]
			print("Number of samples in the corners is {0} out of {1}.\n".format(len(inds_corner),self.N0))

			# plot samples
			# fig, ax = plt.subplots()
			# ax.plot(xs,xis, 'o')
			# plt.show()

			y_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			t_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			x_tf = tf.constant(self.X0[:,[0]],dtype = tf.float32)
			xi_tf = tf.constant(self.X0[:,[1]],dtype = tf.float32)
			target_tf = tf.constant(self.u0, dtype = tf.float32)
			N = tf.constant(self.N0, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
			self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weight':weight}
			samples_list.append(self.X0_dict)
		return samples_list

	def generate_PINN_tests(self):
		N = self.N0_tests.shape[0]
		y_tf = tf.constant((),shape = (N,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (N,0),dtype = tf.float32)
		x_tf = tf.constant(self.N0_tests[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.N0_tests[:,[1]],dtype = tf.float32)
		target_tf = tf.constant(self.u0_tests, dtype = tf.float32)
		N = tf.constant(N, dtype = tf.float32)
		weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weight':weight}
		return X0_dict, target_tf

	@tf.function
	def f_res(self, x_tf, y_tf, t_tf, xi_tf, u, u_x, u_y, u_t, u_xx, u_yy):
		f_u = (u_xx+u_x+xi_tf)/(xi_tf)
		# f_u = (u_xx+u_x+xi_tf)/xi_tf
		# f_u = u_xx+u_x+xi_tf
		return f_u

	@tf.function
	def neumann_bc(self, x_tf, y_tf, t_tf, xi_tf, u_x, u_y):
		# return -u_x/xi_tf
		# return -u_x
		# return -u_x*xi_tf 
		return -u_x/tf.math.sqrt(xi_tf)

	def test_NN(self, net, record_path = None, save_name = None):
		if record_path is not None:
			folderpath = record_path
			record_path = record_path + "rel_errs2.csv"
			if os.path.exists(record_path):
				pass
			else:
				with open(record_path, mode='w') as record:
					fields=['Problem','Net_struct','Net_setup','Sample','L','relative_err','data_name']
					record_writer = csv.writer(record, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					record_writer.writerow(fields)
		X0_dict, u_test = self.u_exact_test()
		x_tf = X0_dict["x_tf"]
		y_tf = X0_dict["y_tf"]
		t_tf = X0_dict["t_tf"]
		xi_tf = X0_dict["xi_tf"]
		target_f = tf.zeros([self.N*self.N_p_test,1])
		if self.sampling_method == 3:
			net.h_init = tf.constant(self.h_init,dtype = tf.float32)
		u_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf)
		f_res = net.compute_residual(x_tf, y_tf, t_tf, xi_tf, target_f)

		neumann_N = 1001
		xn_tf = tf.constant(np.zeros([neumann_N,1]),shape = (neumann_N,1),dtype = tf.float32)
		yn_tf = tf.constant((),shape = (neumann_N,0),dtype = tf.float32)
		tn_tf = tf.constant((),shape = (neumann_N,0),dtype = tf.float32)
		xin_tf = tf.constant(np.power(10,np.linspace(-4,0,neumann_N)),shape = (neumann_N,1),dtype = tf.float32)
		targetn_tf = tf.zeros([neumann_N,1])
		neumann = net.compute_neumann(xn_tf, yn_tf, tn_tf, xin_tf, targetn_tf)
		fig, ax = plt.subplots()
		ax.plot(np.log10(xin_tf.numpy()), neumann.numpy(), 'o', color ="red")
		ax.set_xlabel(r'$\epsilon$')
		ax.set_ylabel(r'Neumann')
		plt.show()

		if self.sampling_method == 0:
			u_test_p = u_test_p.numpy()
			self.V = np.load(self.path_env+"V_{}.npy".format(self.L))
			u_test_p = u_test_p@self.V.T
			u_test_p_grid = tf.constant(u_test_p, dtype = tf.float32)
			u_test_grid = tf.constant(u_test.T, dtype = tf.float32)

		elif self.sampling_method == 1 or self.sampling_method == 2:
			N_record = [self.Nf, self.Nb, self.Nn, self.N0]
			u_test_grid = tf.reshape(u_test,(self.N_p_test,self.N))
			u_test_p_grid = tf.reshape(u_test_p,(self.N_p_test,self.N))
			f_res_grid = tf.reshape(f_res, (self.N_p_test,self.N))

		if self.sampling_method == 3:
			u_test_p = u_test_p.numpy()
			u_test_p_grid = tf.constant(u_test_p, dtype = tf.float32)
			u_test_grid = tf.constant(u_test.T, dtype = tf.float32)
			f_res_grid = None
			
		err_grid = u_test_grid-u_test_p_grid
		err_test = tf.math.reduce_mean(tf.square(err_grid))

		relative_err_vec = tf.norm(err_grid,axis=1)/tf.norm(u_test_grid,axis=1)
		rel_err_test = tf.reduce_mean(relative_err_vec)
		if record_path is not None:
			# y_tf = tf.constant((),shape = (len(self.x),0),dtype = tf.float32)
			# t_tf = tf.constant((),shape = (len(self.x),0),dtype = tf.float32)
			# x_tf = tf.constant(self.x.reshape((len(self.x),1)),dtype = tf.float32)
			# xi_tf = tf.constant(1e-4*np.ones((len(self.x),1)),dtype = tf.float32)
			# u_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf)
			list_info = [self.name,net.name, net.layers,N_record,self.L,rel_err_test.numpy(),save_name]
			# scipy.io.savemat(folderpath+"/{0}.mat".format(N_record), {'approx':u_test_p.numpy()})
			with open(record_path, 'a') as f:
				writer = csv.writer(f)
				writer.writerow(list_info)
		print("Test average error is: {0}\nRelative error is: {1}".format(err_test.numpy(), rel_err_test.numpy()))

		return u_test_grid, u_test_p_grid, err_test, rel_err_test, f_res_grid

	def plot_NN(self, net, figure_save_path = None):
		u_test_grid, u_test_p_grid, _, _, f_res_grid = self.test_NN(net, None)
		# if not os.path.exists(figure_save_path):
		    # os.makedirs(figure_save_path)
		for i in range(0,self.N_p_test):
			xi = self.mu_mat_test[0,i]
			u_test_i = u_test_grid[i].numpy()
			u_test_p_i = u_test_p_grid[i].numpy()

			# if figure_save_path is not None:
			# 	folder_path = "{1}xi_{0}".format(xi, figure_save_path)
			# 	if not os.path.exists(folder_path):
			# 	    os.makedirs(folder_path)
			# 	scipy.io.savemat(folder_path+"/data4.mat", {'true_solution':u_test_i, 'approximation': u_test_p_i, 'xi':xi, 'x':self.x})

			fig, ax = plt.subplots()
			ax.plot(self.x, u_test_p_i, 'o', color ="red")
			ax.plot(self.x, u_test_i)
			ax.set_xlabel(r'$x$')
			ax.set_ylabel(r'$u$')
			fig.suptitle(r"$\xi$ = {0}".format(xi))
			# if figure_save_path is not None:
				# plt.savefig("{1}/u_xi_{0}.png".format(xi,folder_path))
				# plt.cla()
				# plt.clf()
				# plt.close()
			# else:
			# plt.ioff()
			# plt.gcf().show()
			plt.show()

	def select_region(self, inputs, vec, n, epsilon):
		vec_mean = np.mean(vec)
		vec_std = np.std(vec)
		inds = [i for i in range(len(vec)) if vec[i]>(n*vec_std+vec_mean) or vec[i]<(-n*vec_std+vec_mean)]

		inputs_sel = inputs[inds]
		inputs_sel_range = []
		for i in range(inputs_sel.shape[1]):
			imax = np.max(inputs_sel[:,i])
			imin = np.min(inputs_sel[:,i])
			inputs_sel_range.append([np.max([imin-epsilon,self.x_p_domain[i][0]]),np.min([imax+epsilon,self.x_p_domain[i][1]])])
		return inputs_sel_range




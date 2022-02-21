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

class NS_AL2:
	def __init__(self, N_p_train, N_p_test, h, type_weighting = [1,1,1,1], inner = False, sampling_method = 0, add_sample = False, path_env = "./Environments/", L = 0):
		self.name = "NS_AL"
		self.sampling_method = sampling_method
		self.u_dim = 2
		self.P_dim = 3
		self.domain = np.array([[-1,1],[-1,1]])
		self.plimits = np.array([[-4,0]])
		self.x_p_domain = np.array([[-1, 1], [-1, 1], [-4, 0]])
		self.h = h
		self.path_env = path_env
		self.type_weighting = type_weighting
		self.L = L

		# return full solutions for parameters generated
		if self.sampling_method == 0:
			self.lb = np.array([1e-4])
			self.ub = np.array([1.0])
			# self.lb = np.array([-4])
			# self.ub = np.array([0])
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
			self.state_space_size = self.u_dim+self.P_dim+2
			self.output_space_size  = 2
			self.Ns = N_p_train
			self.lb = np.array([-1.0, -1.0, 1e-4, -2.0, 0.0,0.0,0.0])
			self.ub = np.array([1.0, 1.0, 1.0, -2e-4, 1.0,1.0,1.0])
			self.Nf = N_p_train[0]
			self.Nb = N_p_train[1]
			self.Nn = N_p_train[2]
			self.N0 = N_p_train[3]
			if len(N_p_train)>4:
				self.Nr = N_p_train[4]
			else:
				self.Nr = 0

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
		self.N = int(2/self.h)+1

		self.nx = self.N-2
		self.ny = self.N-2

		self.x = np.linspace(self.domain[0,0],self.domain[0,1],num=self.N)
		self.y = np.linspace(self.domain[1,0],self.domain[1,1],num=self.N)
		self.x_in = self.x[1:-1]
		self.y_in = self.y[1:-1]

		self.X, self.Y = np.meshgrid(self.x,self.y)
		self.X_in = self.X[1:-1,1:-1]
		self.Y_in = self.Y[1:-1,1:-1]

		self.inner = inner


	def generate_para(self,app_str = ""):
		np.random.seed(10)
		sampling = LHS(xlimits=self.plimits)
		# sampling = LHS(xlimits=np.array([[-2,-1]]))
		self.app_str = app_str
		filename = self.path_env+"NS_AL_{0}{1}.npy".format(self.N_p_train,app_str)
		
		# check if train parameters exist
		if os.path.exists(filename):
			self.mu_mat_train = np.load(filename)
		else: 
			self.mu_mat_train = sampling(self.N_p_train).T
			self.mu_mat_train[0,:] = np.power(10,self.mu_mat_train[0,:])
			np.save(filename,self.mu_mat_train)

	def generate_para_test(self,app_str = ""):
		np.random.seed(10)
		sampling = LHS(xlimits=self.plimits)
		# sampling = LHS(xlimits=np.array([[-2,-1]]))

		# check if test parameters exist
		if os.path.exists(self.path_env+"NS_AL_{0}{1}.npy".format(self.N_p_test,app_str)):
			self.mu_mat_test = np.load(self.path_env+"NS_AL_{0}{1}.npy".format(self.N_p_test,app_str))
		else:
			self.mu_mat_test = sampling(self.N_p_test).T
			self.mu_mat_test[0,:] = np.power(10,self.mu_mat_test[0,:])
			np.save(self.path_env+"NS_AL_{0}{1}.npy".format(self.N_p_test,app_str),self.mu_mat_test)

	def u_exact_train(self,app_str = ""):
		if self.sampling_method == 0:
			self.u_samples = self.use_ifiss(self.N_p_train,app_str)
			# for i in range(self.mu_mat_train.shape[1]):
			# 	p = self.mu_mat_train[:,i]
			# 	self.generate_one_sol(p)
			return [self.mu_mat_train.T, self.u_samples]
		elif self.sampling_method == 3:
			return self.generate_RNN_samples()

	def use_ifiss(self,num,app_str=""):
		if os.path.exists(self.path_env+"CD_2D_{0}{2}_{1}.mat".format(num,self.N,app_str)):
			u = scipy.io.loadmat(self.path_env+"CD_2D_{0}{2}_{1}.mat".format(num,self.N,app_str))
			u = np.array(u.get('us'))
		else:
			print("run matlab for {0} and {1} grid. L={2}.".format(num,self.N,self.L))
			print("name:","CD_2D_{0}{2}_{1}.mat".format(num,self.N,app_str))
			input()
			u = scipy.io.loadmat(self.path_env+"CD_2D_{0}{2}_{1}.mat".format(num,self.N,app_str))
			u = np.array(u.get('us'))
		return u

	def u_exact_test(self,app_str = ""):
		self.generate_para_test(app_str)
		if self.sampling_method == 0:
			self.u_tests = self.use_ifiss(self.N_p_test,app_str)
			# self.u_tests = np.array([])
			# for i in range(self.mu_mat_test.shape[1]):
			# 	p = self.mu_mat_test[:,i]
			# 	self.generate_one_sol(p, test = True)			
			return self.generate_POD_tests()
		elif self.sampling_method == 1 or self.sampling_method == 2:
			self.N0_tests = np.array([])
			self.u0_tests = np.array([])
			for i in range(self.mu_mat_test.shape[1]):
				p = self.mu_mat_test[:,i]
				self.generate_one_sol(p, test = True)
			# self.u0_tests = self.use_ifiss(self.mu_mat_test.shape[1],"_SD_rand2")
			return self.generate_PINN_tests()
		elif self.sampling_method == 3:
			self.u_tests = np.array([])
			for i in range(self.mu_mat_test.shape[1]):
				p = self.mu_mat_test[:,i]
				self.generate_one_sol(p, test = True)
			return self.generate_RNN_tests()

	def u_exact(self, X):
		x = X[:,[0]]
		y = X[:,[1]]
		xi = X[:,[2]]
		u = x*((1-np.exp((y-1)/xi))/(1-np.exp(-2/xi)))
		# u = 1-np.exp((x-1)/xi)
		return u

	def generate_A_F(self,p):
		xi0 = p[0]
		# xi1 = self.p[1]
		# xi2 = self.p[2]
		# xi_tot = 1
		tot_dim = self.nx*self.ny

		I = sparse.eye(self.nx)
		Iy = sparse.eye(self.ny)
		e1 = np.ones((self.nx))
		e2 = e1.copy()
		e2[-1] = 2
		e3 = e1.copy()
		e3[-1] = 0
		T = sparse.diags([1,-4,1],[-1,0,1],shape=(self.nx,self.nx))
		S = sparse.diags([e2,e1],[-1,1],shape=(self.ny,self.ny))
		S1 = sparse.diags([-e3,e1],[-1,1],shape=(self.ny,self.ny))
		S2 = sparse.diags([-1,1],[-1,1],shape=(self.nx,self.nx))

		A = sparse.kron(Iy,T)+sparse.kron(S,I)
		B = sparse.kron(S1,I)
		C = sparse.kron(Iy,S2)

		Dy_diag = np.sqrt(3)/2*np.ones((self.X_in.shape))
		Dx_diag = -1/2*np.ones((self.Y_in.shape))
		Dy = sparse.diags(Dy_diag.flatten(),0,shape = (tot_dim,tot_dim))
		Dx = sparse.diags(Dx_diag.flatten(),0,shape = (tot_dim,tot_dim))

		L = -xi0*A/(self.h**2)+Dy@B/(2*self.h)+Dx@C/(2*self.h)

		v1 = np.zeros((tot_dim,1))
		v2 = np.zeros((tot_dim,1))
		v3 = np.zeros((tot_dim,1))
		v4 = np.zeros((tot_dim,1))
		v1[int(self.nx/2):self.nx] = 1 # lower
		v2[range(0,tot_dim,self.nx),0] = 0 #left
		v3[range(self.nx-1,tot_dim,self.nx),0] = 1 #right 
		v4[-self.nx:-1] = 0 #upper

		F = xi0*(v1+v2+v3+v4)/(self.h**2)+Dy@(v1-v4)/(2*self.h)+Dx@(v2-v3)/(2*self.h)
		return L,F

	def generate_one_sol(self, p, test = False):
		# L,F = self.generate_A_F(p)
		# u = spsolve(L,F)
		# u = u.reshape((self.ny, self.nx))
		u1 = 1-(self.Y**2).flatten()
		u2 = np.zeros((self.N**2,1))
		self.compile_output(u1, u2, p, test)
				
	def compile_output(self, u1, u2, p, test):
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
				# u_tol = self.fill_BC(u,p)
				u1_tol = u1.reshape((self.N*self.N,1))
				u2_tol = u2.reshape((self.N*self.N,1))
				u0_tol = np.concatenate((u1_tol,u2_tol),axis = 1)
				self.N0_tests = np.concatenate((self.N0_tests, X_0),axis = 0) if self.N0_tests.size else X_0
				self.u0_tests = np.concatenate((self.u0_tests, u0_tol),axis = 0) if self.u0_tests.size else u0_tol								
				
	def fill_BC(self,inner_u,p):
		u_temp = inner_u.reshape((self.ny,self.nx))
		U = np.zeros((self.N,self.N))
		U[1:-1,1:-1] = u_temp
		U[:,0] = 0 # left
		U[:,-1] = 1 # right
		U[-1,:] = 0 # upper
		U[0,:] = 0 # lower
		U[0,int(self.nx/2)::] = 1
		return U

	def create_inner_X(self, p):
		X_in = self.X_in
		Y_in = self.Y_in
		X = X_in.reshape(((self.ny)*self.nx,1))
		Y = Y_in.reshape(((self.ny)*self.nx,1))
		P = p*np.ones(((self.ny)*self.nx,self.P_dim))
		X_f = np.concatenate((X,Y,P),axis=1)
		return X_f

	def create_tol_X(self,p):
		X = self.X.reshape((self.N*self.N,1))
		Y = self.Y.reshape((self.N*self.N,1))
		P = p*np.ones((self.N*self.N,self.P_dim))
		X_f = np.concatenate((X,Y,P),axis=1)
		w1 = 1-Y**2
		w2 = Y*0
		t1 = -2*P
		t2 = Y*0
		X_f = np.concatenate((X_f,w1,w2,t1,t2),axis=1)
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
		self.app_str = app_str
		p_train, u_train = self.u_exact_train(app_str)
		if os.path.exists(self.path_env+"{2}_{1}{3}_V_{0}.npy".format(self.L, self.N_p_train, self.name,app_str)):
			self.V = np.load(self.path_env+"{2}_{1}{3}_V_{0}.npy".format(self.L, self.N_p_train, self.name,app_str))
		else:
			u,s,v = np.linalg.svd(u_train)
			self.V = u[:,0:self.L]
			np.save(self.path_env+"{2}_{1}{3}_V_{0}.npy".format(self.L, self.N_p_train, self.name,app_str),self.V)
		# p_batch = np.log10(p_train)
		p_batch = p_train
		u_batch = u_train
		# print(u_batch[:,0])
		# print(u_batch.shape)
		# input()
		u_batch = self.V.T@u_batch
		# print(u_batch)
		# print(u_batch.shape)
		# input()
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
		# xi_tf = tf.constant(np.log10(self.mu_mat_test).T ,dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		x_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		N = tf.constant(self.N_p_test, dtype = tf.float32)
		weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'N':N, 'type':'Init', 'weight':weight}
		return X0_dict, self.u_tests

	def generate_PINN_samples(self, app_str = ""):
		self.app_str = app_str
		samples_list = []

		filename = "NS_AL2_{0}{1}.npz".format(self.Ns,app_str)
		if os.path.exists("{1}{0}".format(filename,self.path_env)):
			npzfile = np.load("{1}{0}".format(filename,self.path_env))
			if self.Nf>0:
				self.Xf = npzfile['Xf']
				target_f = npzfile['target_f']
			else:
				self.Xf_tf = None
			if self.Nb>0:
				self.Xb_d = npzfile['Xb_d']
				self.ub_d = npzfile['ub_d']

			if self.N0>0:
				self.X0 = npzfile['X0']
				self.u0 = npzfile['u0']
			else:
				self.X0_tf = None
				self.u0_tf = None
			if self.Nn>0:
				self.Xn = npzfile['Xn']
				self.un = npzfile['un']
			else:
				self.Xn = None 
				self.un = None
			if self.Nr>0:
				self.Xr = npzfile['Xr']
				target_r = npzfile['target_r']
			else:
				self.Xr_tf = None
		else:
			np.random.seed(10)

			# sampling_f = LHS(xlimits = np.array([[-1, 1], [-1, 1], [-4, 0]]))
			sampling_f = LHS(xlimits = self.x_p_domain)
			self.Xf = sampling_f(self.Nf)
			self.Xf[:,2] = np.power(10, self.Xf[:,2])
			w1 = 1-self.Xf[:,[1]]**2
			w2 = 0*self.Xf[:,[1]]
			t1 = -2*self.Xf[:,[2]]
			t2 = 0*self.Xf[:,[2]]
			self.Xf = np.concatenate((self.Xf,w1,w2,t1,t2),axis = 1)
			target_f = np.zeros((self.Nf,2))

			sampling_b = LHS(xlimits = np.array([[-1, 1], [-4, 0]]))
			Nb_side = self.Nb//3
			x_p_b = sampling_b(Nb_side)

			pb = x_p_b[:,[1]]
			pb_10= np.power(10, pb)
			xyb = x_p_b[:,[0]]

			lb = np.concatenate((-np.ones((Nb_side,1)),xyb,pb_10,1-xyb**2,np.zeros((Nb_side,1)),-2*pb_10,np.zeros((Nb_side,1))),axis = 1)
			ulb = np.zeros((Nb_side,2))
			# rb = np.concatenate((np.ones([Nb_side,1]),xyb,pb_10,1-xyb**2,np.zeros((Nb_side,1))),axis = 1)
			# urb = np.zeros((Nb_side,2))
			tb = np.concatenate((xyb,np.ones((Nb_side,1)),pb_10,np.zeros((Nb_side,1)),np.zeros((Nb_side,1)),-2*pb_10,np.zeros((Nb_side,1))),axis = 1)
			utb = np.zeros((Nb_side,2))
			db = np.concatenate((xyb,-np.ones((Nb_side,1)),pb_10,np.zeros((Nb_side,1)),np.zeros((Nb_side,1)),-2*pb_10,np.zeros((Nb_side,1))),axis = 1)
			udb = np.zeros((Nb_side,2))

			# self.Xb_d = np.concatenate((lb,rb,tb,db),axis = 0)
			# self.ub_d = np.concatenate((ulb,urb,utb,udb),axis = 0)

			self.Xb_d = np.concatenate((lb,tb,db),axis = 0)
			self.ub_d = np.concatenate((ulb,utb,udb),axis = 0)

			if self.Nn>0:
				# if app_str == "_altneumann":
				sampling_n = LHS(xlimits = np.array([[-1, 1], [-4, 0]]))
				x_p_n = sampling_n(self.Nn)

				pn = x_p_n[:,[1]]
				pn_10= np.power(10, pn)
				xyn = x_p_n[:,[0]]
				rb = np.concatenate((np.ones([self.Nn,1]),xyn,pn_10,1-xyn**2,np.zeros((Nb_side,1)),-2*pb_10,np.zeros((self.Nn,1))),axis = 1)
				urb = np.zeros((self.Nn,2))
				self.Xn = rb 
				self.un = urb
				# else:
				# 	sampling_n = LHS(xlimits = np.array([[-1, 1], [-4, 0]]))
				# 	x_p_n = sampling_n(self.Nn)

				# 	pn = x_p_n[:,[1]]
				# 	pn_10= np.power(10, pn)
				# 	xyn = x_p_n[:,[0]]
				# 	rb = np.concatenate((np.ones([self.Nn,1]),xyn,pn_10,-2*pb_10,np.zeros((self.Nn,1))),axis = 1)
				# 	urb = np.zeros((self.Nn,2))
				# 	urb[:, [0]] = -2*pn_10*np.ones([self.Nn,1])
				# 	self.Xn = rb 
				# 	self.un = urb
			else:
				self.Xn = None 
				self.un = None 

			if self.Nr>0:
				sampling_r = LHS(xlimits = self.x_p_domain)
				self.Xr = sampling_f(self.Nr)
				self.Xr[:,2] = np.power(10, self.Xr[:,2])
				w1 = -2*self.Xf[:,[2]]
				w2 = 0*self.Xf[:,[2]]
				self.Xr = np.concatenate((self.Xr,w1,w2),axis = 1)
				target_r = np.zeros((self.Nr,1))

			else:
				self.Xr = None
				target_r = None

			np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, target_f = target_f, Xb_d = self.Xb_d, ub_d = self.ub_d, Xn = self.Xn, un = self.un, Xr = self.Xr, target_r = target_r)
			# if self.Nr>0:
			# 	sampling_r = LHS(xlimits = self.x_p_domain)
			# 	self.Xr = sampling_f(self.Nr)
			# 	self.Xr[:,2] = np.power(10, self.Xr[:,2])
			# 	target_r = np.zeros([self.Nr,1])
			# else:
			# 	self.Xr = None

			# if self.N0>0:
			# 	sampling_0 = LHS(xlimits = self.x_p_domain)
			# 	x = sampling_0(self.N0)
			# 	x[:,2] = np.power(10, x[:,2])

			# 	str_arr = app_str.split("_")
			# 	if len(str_arr)==3:
			# 		setting_str = str_arr[1]
			# 		setting_para_str = str_arr[2]
			# 		setting_para_str = setting_para_str.replace("p",".")
			# 		setting = int(setting_str)
			# 		setting_para = float(setting_para_str)
			# 	else:
			# 		setting = 0
			# 	# setting 1
			# 	if setting == 1:
			# 		x[:,0] = x[:,1]*x[:,0]+1-x[:,1]
			# 	# setting 2
			# 	elif setting == 2:
			# 		x[:,0] = setting_para*x[:,1]*x[:,0]+1-setting_para*x[:,1]
			# 		x[:,0] = np.maximum(0.001*np.ones(np.shape(x[:,0])),x[:,0])
			# 	# setting 3
			# 	elif setting == 3:
			# 		percent = setting_para/100
			# 		in_corner_int = int(percent*self.N0)
			# 		out_corner_int = self.N0-in_corner_int
			# 		N0_sel = np.random.choice(self.N0,in_corner_int, replace=False)
			# 		N0_sel_diff = np.array([i for i in range(self.N0) if (i not in N0_sel)])
			# 		N0_sel = N0_sel.reshape([in_corner_int,1])
			# 		N0_sel_diff = N0_sel_diff.reshape([out_corner_int,1])
			# 		x[N0_sel,0] = x[N0_sel,1]*x[N0_sel,0]+1-x[N0_sel,1]
			# 		x[N0_sel_diff,0] = (1-x[N0_sel_diff,1])*x[N0_sel_diff,0]

			# 	# self.X0 = np.concatenate((x,1e-4*np.ones((self.N0,1))),axis = 1)
			# 	self.X0 = x
			# 	if app_str == "_reduced":
			# 		leftinds = self.X0[:,1]+np.sqrt(3)*self.X0[:,0]<-1
			# 		self.u0 = np.ones((self.N0,1))
			# 		self.u0[leftinds] = 0
			# 	else:
			# 		self.u0 = self.u_exact(self.X0)
			# 	np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, X0 = self.X0, u0 = self.u0, Xr = self.Xr)
			# else:
			# np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, target_f = target_f, Xb_d = self.Xb_d, ub_d = self.ub_d, Xn = self.Xn, un = self.un)
		
		if self.Nf>0:
			t_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
			x_tf = tf.constant(self.Xf[:,[0]],dtype = tf.float32)
			y_tf = tf.constant(self.Xf[:,[1]],dtype = tf.float32)
			xi_tf = tf.constant(self.Xf[:,[2]],dtype = tf.float32)
			w1_tf = tf.constant(self.Xf[:,[3]],dtype = tf.float32)
			w2_tf = tf.constant(self.Xf[:,[4]],dtype = tf.float32)
			t1_tf = tf.constant(self.Xf[:,[5]],dtype = tf.float32)
			t2_tf = tf.constant(self.Xf[:,[6]],dtype = tf.float32)
			target_tf = tf.constant(target_f, dtype = tf.float32)
			N = tf.constant(self.Nf, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[0], dtype = tf.float32)
			self.Xf_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'w1_tf':w1_tf, 'w2_tf':w2_tf, 't1_tf':t1_tf, 't2_tf':t2_tf, 'target':target_tf, 'N':N, 'type':'Res', 'weight':weight}
			samples_list.append(self.Xf_dict)

		if self.Nb>0:
			t_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
			x_tf = tf.constant(self.Xb_d[:,[0]],dtype = tf.float32)
			y_tf = tf.constant(self.Xb_d[:,[1]],dtype = tf.float32)
			xi_tf = tf.constant(self.Xb_d[:,[2]],dtype = tf.float32)
			w1_tf = tf.constant(self.Xb_d[:,[3]],dtype = tf.float32)
			w2_tf = tf.constant(self.Xb_d[:,[4]],dtype = tf.float32)
			t1_tf = tf.constant(self.Xb_d[:,[5]],dtype = tf.float32)
			t2_tf = tf.constant(self.Xb_d[:,[6]],dtype = tf.float32)
			target_tf = tf.constant(self.ub_d, dtype = tf.float32)
			N = tf.constant(self.Nb, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[1], dtype = tf.float32)
			self.Xb_d_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'w1_tf':w1_tf, 'w2_tf':w2_tf, 't1_tf':t1_tf, 't2_tf':t2_tf, 'target':target_tf, 'N':N, 'type':'B_D', 'weight':weight}
			samples_list.append(self.Xb_d_dict)

		if self.Nn>0:
			t_tf = tf.constant((),shape = (self.Nn,0),dtype = tf.float32)
			x_tf = tf.constant(self.Xn[:,[0]],dtype = tf.float32)
			y_tf = tf.constant(self.Xn[:,[1]],dtype = tf.float32)
			xi_tf = tf.constant(self.Xn[:,[2]],dtype = tf.float32)
			w1_tf = tf.constant(self.Xn[:,[3]],dtype = tf.float32)
			w2_tf = tf.constant(self.Xn[:,[4]],dtype = tf.float32)
			t1_tf = tf.constant(self.Xn[:,[5]],dtype = tf.float32)
			t2_tf = tf.constant(self.Xn[:,[6]],dtype = tf.float32)
			target_tf = tf.constant(self.un, dtype = tf.float32)
			N = tf.constant(self.Nn, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[2], dtype = tf.float32)
			self.Xb_n_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'w1_tf':w1_tf, 'w2_tf':w2_tf,'t1_tf':t1_tf, 't2_tf':t2_tf, 'target':target_tf, 'N':N, 'type':'B_N', 'weight':weight}
			samples_list.append(self.Xb_n_dict)

		if self.N0>0:
			#check number of samples in (1-xi,1) corner
			# xis = self.X0[:,[2]]
			# xs = self.X0[:,[0]]
			# inds_corner = [i for i in range(len(xs)) if xs[i]>=1-xis[i]]
			# print("Number of samples in the corners is {0} out of {1}.\n".format(len(inds_corner),self.N0))

			# plot samples
			# fig, ax = plt.subplots()
			# ax.plot(xs,xis, 'o')
			# plt.show()

			# y_tf = tf.constant(self.X0[:,[1]],dtype = tf.float32)
			# t_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			# x_tf = tf.constant(self.X0[:,[0]],dtype = tf.float32)
			# xi_tf = tf.constant(self.X0[:,[2]],dtype = tf.float32)
			# target_tf = tf.constant(self.u0, dtype = tf.float32)
			# N = tf.constant(self.N0, dtype = tf.float32)
			# weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
			# self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weight':weight}
			# samples_list.append(self.X0_dict)
			pass
		if self.Nr>0:
			t_tf = tf.constant((),shape = (self.Nr,0),dtype = tf.float32)
			x_tf = tf.constant(self.Xr[:,[0]],dtype = tf.float32)
			y_tf = tf.constant(self.Xr[:,[1]],dtype = tf.float32)
			xi_tf = tf.constant(self.Xr[:,[2]],dtype = tf.float32)
			w1_tf = tf.constant(self.Xr[:,[3]],dtype = tf.float32)
			w2_tf = tf.constant(self.Xr[:,[4]],dtype = tf.float32)
			t1_tf = tf.constant(self.Xr[:,[5]],dtype = tf.float32)
			t2_tf = tf.constant(self.Xr[:,[6]],dtype = tf.float32)
			target_tf = tf.constant(target_r, dtype = tf.float32)
			N = tf.constant(self.Nr, dtype = tf.float32)
			weight = tf.constant(self.type_weighting[4], dtype = tf.float32)
			self.Xr_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'w1_tf':w1_tf, 'w2_tf':w2_tf,'t1_tf':t1_tf, 't2_tf':t2_tf, 'target':target_tf, 'N':N, 'type':'Div', 'weight':weight}
			samples_list.append(self.Xr_dict)
			# pass

		return samples_list

	def generate_PINN_tests(self):
		N = self.N0_tests.shape[0]
		t_tf = tf.constant((),shape = (N,0),dtype = tf.float32)
		x_tf = tf.constant(self.N0_tests[:,[0]],dtype = tf.float32)
		y_tf = tf.constant(self.N0_tests[:,[1]],dtype = tf.float32)
		xi_tf = tf.constant(self.N0_tests[:,[2]],dtype = tf.float32)
		w1_tf = tf.constant(self.N0_tests[:,[3]],dtype = tf.float32)
		w2_tf = tf.constant(self.N0_tests[:,[4]],dtype = tf.float32)
		t1_tf = tf.constant(self.N0_tests[:,[5]],dtype = tf.float32)
		t2_tf = tf.constant(self.N0_tests[:,[6]],dtype = tf.float32)
		target_tf = tf.constant(self.u0_tests, dtype = tf.float32)
		N = tf.constant(N, dtype = tf.float32)
		weight = tf.constant(self.type_weighting[3], dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'w1_tf':w1_tf, 'w2_tf':w2_tf, 't1_tf':t1_tf, 't2_tf':t2_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weight':weight}
		return X0_dict, target_tf

	@tf.function
	def f_res(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf,t2_tf, u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy):
		f_u1 = -xi_tf*(u1_xx+u1_yy)+w1_tf*u1_x+w2_tf*u1_y+u1_xx+u2_xy+t1_tf
		f_u2 = -xi_tf*(u2_xx+u2_yy)+w1_tf*u2_x+w2_tf*u2_y+u2_yy+u1_xy+t2_tf
		return f_u1, f_u2

	@tf.function 
	def lhs_res(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf,t2_tf,u, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy):
		pass
		# f_u = -xi_tf*u_xx-xi_tf*u_yy-u_x/2+tf.math.sqrt(3.0)/2*u_y
		# return f_u

	@tf.function
	def f_reduced_res(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf,t1_tf,t2_tf, u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy):
		pass
		# f_u = -u_x/2+tf.math.sqrt(3.0)/2*u_y
		# f_u = (-u_xx*xi_tf+u_x-1)/xi_tf
		# return f_u

	@tf.function
	def neumann_bc(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf,t2_tf,u1_x, u1_y, u2_x, u2_y):
		return xi_tf*u1_x, u2_x

	def test_NN(self, net, record_path = None,save_name = None):
		
		if record_path is not None:
			folderpath = record_path
			record_path = record_path + "rel_errs2.csv"
			if os.path.exists(record_path):
				pass
			else:
				with open(record_path, mode='w') as record:
					fields=['Problem','Net_struct','Net_setup','Sample','L','relative_err','save_name','Projection Error','Net Error', 'Error']
					record_writer = csv.writer(record, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					record_writer.writerow(fields)

		X0_dict, u_test = self.u_exact_test(self.app_str)
		u1_test,u2_test = tf.split(u_test,2,axis=1)
		x_tf = X0_dict["x_tf"]
		y_tf = X0_dict["y_tf"]
		t_tf = X0_dict["t_tf"]
		xi_tf = X0_dict["xi_tf"]
		w1_tf = X0_dict["w1_tf"]
		w2_tf = X0_dict["w2_tf"]
		t1_tf = X0_dict["t1_tf"]
		t2_tf = X0_dict["t2_tf"]
		u1_test_p,u2_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf)

		if self.sampling_method == 3:
			net.h_init = tf.constant(self.h_init,dtype = tf.float32)
		if self.sampling_method == 0:
			u_test_p = u_test_p.numpy()
			self.V = np.load(self.path_env+"{2}_{1}{3}_V_{0}.npy".format(self.L, self.N_p_train, self.name, self.app_str))
			
			reduced_sols = self.V.T@u_test

			reduced_err_vec = np.linalg.norm(reduced_sols.T-u_test_p,ord=2,axis=1)
			neterror = np.average(reduced_err_vec)

			# print(reduced_sols.T-u_test_p)
			# input()
			reduced_sols_back = reduced_sols.T@self.V.T
			reduced_sols_back_reshape = tf.reshape(reduced_sols_back,(self.N_p_test,self.N,self.N))
			u_test_p = u_test_p@self.V.T
			u_test_p_grid = tf.constant(u_test_p, dtype = tf.float32)
			u_test_grid = tf.reshape(u_test.T,(self.N_p_test,self.N,self.N))
			u_test_p_grid = tf.reshape(u_test_p,(self.N_p_test,self.N,self.N))
			reduced_error = tf.norm(u_test_grid-reduced_sols_back_reshape,axis=[1,2])/tf.norm(u_test_grid,axis=[1,2])
			reduced_error = tf.reduce_mean(reduced_error)
			print('Reduced error:',reduced_error)
			# input()
			f_res_grid = None


		elif self.sampling_method == 1 or self.sampling_method == 2:
			target_f = tf.zeros([self.N*self.N*self.N_p_test,2])
			u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy = net.derivatives(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf)
			f1_res,f2_res = self.f_res(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy)
			# f_res = net.compute_residual(x_tf, y_tf, t_tf, xi_tf,w1_tf,w2_tf, target_f)
			# f1_res,f2_res = tf.split(f_res,2,axis=1)
			N_record = [self.Nf, self.Nb, self.Nn, self.N0]
			u1_test_grid = tf.reshape(tf.transpose(u1_test),(self.N_p_test,self.N,self.N))
			u2_test_grid = tf.reshape(tf.transpose(u2_test),(self.N_p_test,self.N,self.N))
			u1_test_p_grid = tf.reshape(u1_test_p,(self.N_p_test,self.N,self.N))
			u2_test_p_grid = tf.reshape(u2_test_p,(self.N_p_test,self.N,self.N))
			f1_res_grid = tf.reshape(f1_res, (self.N_p_test,self.N,self.N))
			f2_res_grid = tf.reshape(f2_res, (self.N_p_test,self.N,self.N))
			for i in range(0,self.N_p_test):
				xi = self.mu_mat_test[0,i]
				u1_test_p_i = u1_test_p_grid[i].numpy()
				u2_test_p_i = u2_test_p_grid[i].numpy()
				u1_test_i = u1_test_grid[i].numpy()
				u2_test_i = u2_test_grid[i].numpy()
				err1_i = u1_test_p_i-u1_test_i
				err2_i = u2_test_p_i-u2_test_i
				print(u1_test_p_i)
				fig, ax = plt.subplots()
				ax.quiver(self.X,self.Y,u1_test_p_i,u2_test_p_i)
				ax.set_title(r"NN, $\epsilon$ = {0}".format(xi))
				fig1, ax1 = plt.subplots()
				ax1.quiver(self.X,self.Y,u1_test_i,u2_test_i)
				ax1.set_title(r"Exact, $\epsilon$ = {0}".format(xi))
				fig3, axs = plt.subplots(2)
				cs1 = axs[0].contourf(self.X, self.Y, np.absolute(err1_i))
				axs[0].set_title('u_x')
				cs2 = axs[1].contourf(self.X, self.Y, np.absolute(err2_i))
				axs[1].set_title('u_y')
				fig3.colorbar(cs1, ax = axs[0])
				fig3.colorbar(cs2, ax = axs[1])
				plt.show()


		if self.sampling_method == 3:
			u_test_p = u_test_p.numpy()
			u_test_p_grid = tf.constant(u_test_p, dtype = tf.float32)
			u_test_grid = tf.constant(u_test.T, dtype = tf.float32)
			f_res_grid = None
			
		err_grid = u_test_grid-u_test_p_grid
		err_test = tf.math.reduce_mean(tf.square(err_grid))

		relative_err_vec = tf.norm(err_grid,ord=2,axis=[1,2])/tf.norm(u_test_grid,axis=[1,2],ord=2)
		err_vec = tf.norm(err_grid,axis=[1,2],ord=2)
		err_vec_ave = tf.reduce_mean(err_vec)
		rel_err_test = tf.reduce_mean(relative_err_vec)
		if record_path is not None:
			# y_tf = tf.constant((),shape = (len(self.x),0),dtype = tf.float32)
			# t_tf = tf.constant((),shape = (len(self.x),0),dtype = tf.float32)
			# x_tf = tf.constant(self.x.reshape((len(self.x),1)),dtype = tf.float32)
			# xi_tf = tf.constant(1e-4*np.ones((len(self.x),1)),dtype = tf.float32)
			# u_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf)
			list_info = [self.name,net.name, net.layers, self.N_p_train,self.L,rel_err_test.numpy(),save_name,reduced_error.numpy(),neterror,err_vec_ave.numpy(),np.linalg.norm(self.V.T,ord=2)]
			# scipy.io.savemat(folderpath+"/{0}.mat".format(N_record), {'approx':u_test_p.numpy()})
			with open(record_path, 'a') as f:
				writer = csv.writer(f)
				writer.writerow(list_info)
		print("Test average error is: {0}\nRelative error is: {1}".format(err_test.numpy(), rel_err_test.numpy()))

		if self.sampling_method == 0:
			return u_test_grid, u_test_p_grid, err_test, rel_err_test, f_res_grid,reduced_sols_back_reshape
		else:
			return u_test_grid, u_test_p_grid, err_test, rel_err_test, f_res_grid

	def plot_NN(self, net, figure_save_path = None):
		if self.sampling_method == 0:
			u_test_grid, u_test_p_grid, _, _, f_res_grid, reduced_sols_back_reshape = self.test_NN(net, None)
		else:
			u_test_grid, u_test_p_grid, _, _, f_res_grid = self.test_NN(net, None)

		# if not os.path.exists(figure_save_path):
		    # os.makedirs(figure_save_path)
		for i in range(0,self.N_p_test):
			xi = self.mu_mat_test[0,i]
			u_test_i = u_test_grid[i].numpy()
			u_test_p_i = u_test_p_grid[i].numpy()
			# reduced_sols_i = reduced_sols_back_reshape[i].numpy()

			# if figure_save_path is not None:
			# 	folder_path = "{1}xi_{0}".format(xi, figure_save_path)
			# 	if not os.path.exists(folder_path):
			# 	    os.makedirs(folder_path)
			# 	scipy.io.savemat(folder_path+"/data4.mat", {'true_solution':u_test_i, 'approximation': u_test_p_i, 'xi':xi, 'x':self.x})

			# fig = plt.figure(figsize=plt.figaspect(0.5))
			# ax = fig.add_subplot(1, 2, 1, projection='3d')
			# ax.plot_wireframe(self.X, self.Y, u_test_p_i, color ="red")
			# ax.set_xlabel('x')
			# ax.set_ylabel('y')
			# ax.set_zlabel('u')
			# ax.set_title(r"NN, $\epsilon$ = {0}".format(xi))
			# ax1 = fig.add_subplot(1, 2, 2, projection='3d')
			# ax1.plot_wireframe(self.X, self.Y, u_test_i)
			# ax1.set_xlabel('x')
			# ax1.set_ylabel('y')
			# ax1.set_zlabel('u')
			# ax1.set_title(r"FEM SD, $\epsilon$ = {0}".format(xi))
			fig = plt.figure(1)
			ax = fig.gca(projection='3d')
			ax.plot_wireframe(self.X, self.Y, u_test_p_i, color ="red")
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('u')
			ax.set_title(r"NN, $\epsilon$ = {0}".format(xi))
			fig1 = plt.figure(2)
			ax1 = fig1.gca(projection='3d')
			ax1.plot_wireframe(self.X, self.Y, u_test_i)
			ax1.set_xlabel('x')
			ax1.set_ylabel('y')
			ax1.set_zlabel('u')
			ax1.set_title(r"FEM SD, $\epsilon$ = {0}".format(xi))
			diff = u_test_i-u_test_p_i
			fig2 = plt.figure(3)
			ax2 = fig2.gca(projection='3d')
			ax2.plot_wireframe(self.X, self.Y, diff)
			ax2.set_xlabel('x')
			ax2.set_ylabel('y')
			ax2.set_zlabel('u')
			ax2.set_title(r"Difference, $\epsilon$ = {0}".format(xi))
			# fig = plt.figure(4)
			# ax3 = fig.gca(projection='3d')
			# ax3.plot_wireframe(self.X, self.Y, reduced_sols_i, color ="red")
			# ax3.set_xlabel('x')
			# ax3.set_ylabel('y')
			# ax3.set_zlabel('u')
			# ax3.set_title(r"reduced, $\epsilon$ = {0}".format(xi))
			
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


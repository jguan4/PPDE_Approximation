import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt 
from smt.sampling_methods import LHS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class CD_1D:
	def __init__(self, h, Nf, Nd, N0):
		self.h = h
		self.N = int(1/self.h)+1
		self.nx = self.N-2
		self.X = np.linspace(0,1,num=self.N)
		# number of samples
		self.Nf  = Nf
		self.Nd  = Nd
		self.N0  = N0
		# domain of [x,log(xi)]
		self.x_p_domain = np.array([[0, 1], [-4, 0]])
		# lower and upper bound of [x,xi]
		self.lb = np.array([0.0, 1e-4])
		self.ub = np.array([1.0, 1.0])
		# input and output size of the network
		self.state_space_size = 2
		self.output_space_size  = 1

	def u_exact(self,xi):
		u = 1-np.exp((self.X-1)/xi)
		return u

	def generate_one_sol(self, xi):
		r = self.h/(2*xi)
		A = sparse.diags([-(1+r),2,-(1-r)],[-1,0,1],\
			shape=(self.nx,self.nx), format = 'csr')
		F = np.zeros((self.nx,1))
		F[0] = (1+r)*(1-np.exp(-1/xi))
		u = spsolve(A,F)
		U = np.zeros((self.N,1))
		U[1:-1,0] = u
		U[0] = 1-np.exp(-1/xi)
		return U

	def generate_PINN_samples(self):

		samples_list = []

		np.random.seed(10)

		sampling_f = LHS(xlimits = self.x_p_domain)
		self.Xf = sampling_f(self.Nf)
		self.Xf[:,1] = np.power(10, self.Xf[:,1])
		target_f = np.zeros([self.Nf,1])

		sampling_b = LHS(xlimits = np.array([[-4, 0]]))
		x_p_b = sampling_b(self.Nb//2)
		pb = x_p_b
		pb_10= np.power(10, pb)
		lb = np.concatenate((np.zeros((self.Nb//2,1)),pb_10),axis = 1)
		ulb = 1-np.exp(-1/pb_10)
		rb = np.concatenate((np.ones([self.Nb//2,1]),pb_10),axis = 1)
		urb = np.zeros((self.Nb//2,1))

		self.Xb_d = np.concatenate((lb,rb),axis = 0)
		self.ub_d = np.concatenate((ulb,urb),axis = 0)

		if self.N0>0:
			sampling_0 = LHS(xlimits = )
			x = sampling_0(self.N0)
			self.X0 = np.concatenate((x,1e-4*np.ones((self.N0,1))),axis = 1)
			self.u0 = self.u_exact(self.X0)
			np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, X0 = self.X0, u0 = self.u0)
		else:
			np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d)
		
		y_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
		x_tf = tf.constant(self.Xf[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xf[:,[1]],dtype = tf.float32)
		target_tf = tf.constant(target_f, dtype = tf.float32)
		N = tf.constant(self.Nf, dtype = tf.float32)
		self.Xf_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Res', 'weighting':self.type_weighting[0]}
		samples_list.append(self.Xf_dict)

		y_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
		x_tf = tf.constant(self.Xb_d[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xb_d[:,[1]],dtype = tf.float32)
		target_tf = tf.constant(self.ub_d, dtype = tf.float32)
		N = tf.constant(self.Nb, dtype = tf.float32)
		self.Xb_d_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_D', 'weighting':self.type_weighting[1]}
		samples_list.append(self.Xb_d_dict)

		if self.N0>0:
			y_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			t_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			x_tf = tf.constant(self.X0[:,[0]],dtype = tf.float32)
			xi_tf = tf.constant(self.X0[:,[1]],dtype = tf.float32)
			target_tf = tf.constant(self.u0, dtype = tf.float32)
			N = tf.constant(self.N0, dtype = tf.float32)
			self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weighting':self.type_weighting[3]}
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
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weighting':self.type_weighting[3]}
		return X0_dict, target_tf

	@tf.function
	def f_res(self, x_tf, y_tf, t_tf, xi_tf, u, u_x, u_y, u_t, u_xx, u_yy):
		f_u = -u_xx*xi_tf+u_x
		return f_u

	@tf.function
	def neumann_bc(self, u_x, u_y):
		return

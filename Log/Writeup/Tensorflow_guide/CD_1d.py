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
	def __init__(self, h):
		self.h = h
		self.N = int(1/self.h)+1
		self.nx = self.N-2
		self.X = np.linspace(0,1,num=self.N)
		# domain of [x,log(xi)]
		self.x_p_domain = np.array([[0, 1], [-4, 0]])
		# lower and upper bound of [x,xi]
		self.lb = np.array([0.0, 1e-4])
		self.ub = np.array([1.0, 1.0])
		# input and output size of the network
		self.state_space_size = 2
		self.output_space_size  = 1
		# path to save the samples to
		self.path_env = "./Samples/CD_1D/"
		if not os.path.exists(self.path_env):
			os.makedirs(self.path_env)

	def u_exact(self,X,xi):
		u = 1-np.exp((X-1)/xi)
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

	def generate_PINN_samples(self, Nf, Nd, N0, type_weighting):
		Ns = [Nf,Nd,N0]
		filename = "CD_1D_{0}.npz".format(Ns)

		samples_list = []
		if os.path.exists("{1}{0}".format(filename,self.path_env)):
			npzfile = np.load("{1}{0}".format(filename,self.path_env))
			if Nf>0:
				self.Xf = npzfile['Xf']
				target_f = np.zeros([Nf,1])
			else:
				self.Xf_tf = None
			if Nd>0:
				self.Xb_d = npzfile['Xb_d']
				self.ub_d = npzfile['ub_d']
			if N0>0:
				self.X0 = npzfile['X0']
				self.u0 = npzfile['u0']
			else:
				self.X0_tf = None
				self.u0_tf = None
		else:
			np.random.seed(10)

			sampling_f = LHS(xlimits = self.x_p_domain)
			self.Xf = sampling_f(Nf)
			self.Xf[:,1] = np.power(10, self.Xf[:,1])
			target_f = np.zeros([Nf,1])

			sampling_b = LHS(xlimits = self.x_p_domain[[1]])
			x_p_b = sampling_b(Nd//2)
			pb = x_p_b
			pb_10= np.power(10, pb)
			lb = np.concatenate((np.zeros((Nd//2,1)),pb_10),axis = 1)
			ulb = 1-np.exp(-1/pb_10)
			rb = np.concatenate((np.ones([Nd//2,1]),pb_10),axis = 1)
			urb = np.zeros((Nd//2,1))

			self.Xb_d = np.concatenate((lb,rb),axis = 0)
			self.ub_d = np.concatenate((ulb,urb),axis = 0)

			if N0>0:
				sampling_0 = LHS(xlimits = self.x_p_domain)
				X0 = sampling_0(N0)
				X0[:,1] = np.power(10, X0[:,1])
				self.X0 = X0
				self.u0 = self.u_exact(self.X0[:,[0]],self.X0[:,[1]])
				np.savez("{1}{0}".format(filename,self.path_env), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, X0 = self.X0, u0 = self.u0)
			else:
				np.savez("{1}{0}".format(filename,self.path_env), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d)
		y_tf = tf.constant((),shape = (Nf,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (Nf,0),dtype = tf.float32)
		x_tf = tf.constant(self.Xf[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xf[:,[1]],dtype = tf.float32)
		target_tf = tf.constant(target_f, dtype = tf.float32)
		N = tf.constant(Nf, dtype = tf.float32)
		weight = tf.constant(type_weighting[0], dtype = tf.float32)
		self.Xf_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Res', 'weight':weight}
		samples_list.append(self.Xf_dict)

		y_tf = tf.constant((),shape = (Nd,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (Nd,0),dtype = tf.float32)
		x_tf = tf.constant(self.Xb_d[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xb_d[:,[1]],dtype = tf.float32)
		target_tf = tf.constant(self.ub_d, dtype = tf.float32)
		N = tf.constant(Nd, dtype = tf.float32)
		weight = tf.constant(type_weighting[1], dtype = tf.float32)
		self.Xb_d_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_D', 'weight':weight}
		samples_list.append(self.Xb_d_dict)

		if N0>0:
			y_tf = tf.constant((),shape = (N0,0),dtype = tf.float32)
			t_tf = tf.constant((),shape = (N0,0),dtype = tf.float32)
			x_tf = tf.constant(self.X0[:,[0]],dtype = tf.float32)
			xi_tf = tf.constant(self.X0[:,[1]],dtype = tf.float32)
			target_tf = tf.constant(self.u0, dtype = tf.float32)
			N = tf.constant(N0, dtype = tf.float32)
			weight = tf.constant(type_weighting[2], dtype = tf.float32)
			self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weight':weight}
			samples_list.append(self.X0_dict)
		return samples_list

	def generate_PINN_tests(self, N_test):
		sampling_test = LHS(xlimits = self.x_p_domain[[1]])
		filename = "CD_1D_test_{0}.npz".format(N_test)
		if os.path.exists("{1}{0}".format(filename,self.path_env)):
			npzfile = np.load("{1}{0}".format(filename,self.path_env))
			self.xi_test = npzfile['xi_test']
			xi_test_tol = npzfile['xi_test_tol']
			X_test_tol = npzfile['X_test_tol']
			u_test = npzfile['u_test']
		else:
			xi_test = sampling_test(N_test)
			self.xi_test= np.power(10, xi_test)
			xi_test_tol = self.xi_test*np.ones([1,self.N])
			xi_test_tol = np.reshape(xi_test_tol,(self.N*N_test,1))
			X_test_tol = np.tile(self.X, (N_test,1))
			X_test_tol = np.reshape(X_test_tol, (self.N*N_test,1))
			u_test = self.u_exact(X_test_tol,xi_test_tol)
			np.savez("{1}{0}".format(filename,self.path_env), xi_test = self.xi_test, xi_test_tol = xi_test_tol, X_test_tol = X_test_tol, u_test = u_test)

		y_tf = tf.constant((),shape = (N_test*self.N,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (N_test*self.N,0),dtype = tf.float32)
		x_tf = tf.constant(X_test_tol,dtype = tf.float32)
		xi_tf = tf.constant(xi_test_tol,dtype = tf.float32)
		target_tf = tf.constant(u_test, dtype = tf.float32)
		N = tf.constant(N_test, dtype = tf.float32)
		weight = tf.constant(1, dtype = tf.float32)

		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init", 'weight':weight}
		return X0_dict, target_tf

	@tf.function
	def f_res(self, x_tf, y_tf, t_tf, xi_tf, u, u_x, u_y, u_t, u_xx, u_yy):
		f_u = -u_xx*xi_tf+u_x
		return f_u

	@tf.function
	def neumann_bc(self, u_x, u_y):
		return

	def test_NN(self, net, N_test):
		X0_dict, u_test = self.generate_PINN_tests(N_test)
		x_tf = X0_dict["x_tf"]
		y_tf = X0_dict["y_tf"]
		t_tf = X0_dict["t_tf"]
		xi_tf = X0_dict["xi_tf"]
		u_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf)

		u_test_grid = tf.reshape(u_test,(N_test,self.N))
		u_test_p_grid = tf.reshape(u_test_p,(N_test,self.N))
		err_grid = u_test_grid-u_test_p_grid
		err_test = tf.math.reduce_mean(tf.square(err_grid))

		relative_err_vec = tf.norm(err_grid,axis=1)/tf.norm(u_test_grid,axis=1)
		rel_err_test = tf.reduce_mean(relative_err_vec)

		print("Test average error is: {0}\nRelative error is: {1}".format(err_test.numpy(), rel_err_test.numpy()))

		return u_test_grid, u_test_p_grid, err_test, rel_err_test

	def plot_NN(self, net, N_test):
		u_test_grid, u_test_p_grid, _, _ = self.test_NN(net, N_test)

		for i in range(0,N_test):
			xi = self.xi_test[i,0]
			u_test_i = u_test_grid[i].numpy()
			u_test_p_i = u_test_p_grid[i].numpy()

			fig, ax = plt.subplots()
			ax.plot(self.X, u_test_p_i, color ="red")
			ax.plot(self.X, u_test_i)
			ax.set_xlabel(r'$x$')
			ax.set_ylabel(r'$u$')
			fig.suptitle(r"$\xi$ = {0}".format(xi))
			plt.show()
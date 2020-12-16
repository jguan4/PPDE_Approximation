import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import scipy.io
from smt.sampling_methods import LHS
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import csv   


class Burgers_Equation:
	def __init__(self,N_p_train,N_p_test,h,inner = False, sampling_method = 0, path_env = "./Environments/", L = 0):
		# self.N_p = N_p
		self.name = "Burger"
		self.sampling_method = sampling_method
		self.u_dim = 2
		self.P_dim = 1
		self.inner = inner
		self.h = h #1/(2**7)
		self.domain = np.array([[-1,1],[0,1]])
		self.plimits = np.array([[-3,-2]])
		self.x_p_domain = np.array([[-1, 1], [0, 1], [-3, -2]])
		self.path_env = path_env
		self.L = L
		
		# POD setting
		if self.sampling_method == 0:
			self.lb = np.array([1e-3])
			self.ub = np.array([1e-2])
			self.state_space_size = self.P_dim
			self.output_space_size = None
			self.N_p_train = N_p_train[3]
			self.P_samples = np.array([])
			self.u_samples = np.array([])
			self.u_tests = np.array([])
			self.generate_para()

		elif self.sampling_method == 1 or self.sampling_method == 2:
			self.state_space_size = self.u_dim + self.P_dim
			self.output_space_size  = 1
			self.lb = np.array([-1.0, 0.0, 1e-3])
			self.ub = np.array([1.0, 1.0, 1e-2])
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
			pass

		self.N_p_test = N_p_test
		
		self.N = int(2/self.h)+1
		self.CFL = 1e-4
		self.Nt = 17
		self.X = np.linspace(-1,1,self.N)
		self.T = np.linspace(0,1,self.Nt)

		self.nx = self.N-2
		self.x_in = self.X[1:-1]

	def generate_para(self):
		np.random.seed(10)
		sampling = LHS(xlimits=self.plimits)
		filename = self.path_env+"{1}_{0}.npy".format(self.N_p_train,self.name)
		
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
		if os.path.exists(self.path_env+"{1}_{0}.npy".format(self.N_p_test,self.name)):
			self.mu_mat_test = np.load(self.path_env+"{1}_{0}.npy".format(self.N_p_test,self.name))
		else:
			self.mu_mat_test = sampling(self.N_p_test).T
			self.mu_mat_test[0,:] = np.power(10,self.mu_mat_test[0,:])
			np.save(self.path_env+"{1}_{0}.npy".format(self.N_p_test,self.name),self.mu_mat_test)

	def u_exact_train(self):
		if self.sampling_method == 0:
			if os.path.exists(self.path_env+"{1}_POD_{0}.npy".format(self.N_p_train,self.name)):
				self.u_samples = np.load(self.path_env+"{1}_POD_{0}.npy".format(self.N_p_train,self.name))
			else:
				for i in range(self.mu_mat_train.shape[1]):
					p = self.mu_mat_train[:,i]
					self.generate_one_sol(p)
				np.save(self.path_env+"{1}_POD_{0}.npy".format(self.N_p_train,self.name), self.u_samples)
			return [self.mu_mat_train.T, self.u_samples]
		# elif self.sampling_method == 1 or self.sampling_method == 2:
		# 	return self.generate_PINN_samples()
		elif self.sampling_method == 3:
			return self.generate_RNN_samples()

	def u_exact_test(self):
		self.generate_para_test()
		if self.sampling_method == 0:
			if os.path.exists(self.path_env+"{1}_POD_{0}_test.npy".format(self.N_p_test,self.name)):
				self.u_tests = np.load(self.path_env+"{1}_POD_{0}_test.npy".format(self.N_p_test,self.name))
			else:
				self.u_tests = np.array([])
				for i in range(self.mu_mat_test.shape[1]):
					p = self.mu_mat_test[:,i]
					self.generate_one_sol(p, test = True)
				np.save(self.path_env+"{1}_POD_{0}_test.npy".format(self.N_p_test,self.name), self.u_tests)	
				# return [self.mu_mat_test.T, self.u_tests]
			return self.generate_POD_tests()
		elif self.sampling_method == 1 or self.sampling_method == 2:
			if os.path.exists(self.path_env+"{1}_PINN_{0}_test_input.npy".format(self.N_p_test,self.name)) and os.path.exists(self.path_env+"{1}_PINN_{0}_test_target.npy".format(self.N_p_test,self.name)):
				self.N0_tests = np.load(self.path_env+"{1}_PINN_{0}_test_input.npy".format(self.N_p_test,self.name))
				self.u0_tests = np.load(self.path_env+"{1}_PINN_{0}_test_target.npy".format(self.N_p_test,self.name))
			else:
				self.N0_tests = np.array([])
				self.u0_tests = np.array([])
				for i in range(self.mu_mat_test.shape[1]):
					p = self.mu_mat_test[:,i]
					self.generate_one_sol(p, test = True)
				np.save(self.path_env+"{1}_PINN_{0}_test_input.npy".format(self.N_p_test,self.name), self.N0_tests)	
				np.save(self.path_env+"{1}_PINN_{0}_test_target.npy".format(self.N_p_test,self.name), self.u0_tests)	
			return self.generate_PINN_tests()
		elif self.sampling_method == 3:
			return self.generate_RNN_tests()

	def u_approx(self,p):
		p = p[0]
		h = 1/2**10
		Nx = int(2/h)+1
		k = self.CFL/p*h
		fac = np.ceil(self.T[1]/k)
		k = self.T[1]/fac
		Nt = int(1/k)+1
		X = np.linspace(-1,1,Nx)
		u0 = -np.sin(np.pi*X)
		u0_save = u0[::8]
		us = u0_save.reshape((self.N,1))
		u = u0
		unew = 0*u
		for i in range(1,Nt+1):
			tt = k*i
			fminus = 0.5*( self.f(u[1:-1]) + self.f(u[0:-2]) )
			fplus = 0.5*( self.f(u[1:-1]) + self.f(u[2::]) )
			unew[1:-1] = u[1:-1] + k*(p*(u[2::]-2*u[1:-1]+u[0:-2])/(h**2) - (fplus-fminus)/h )
			unew[0] = u[0]
			unew[-1] = u[-1]
			u = unew
			if i%fac == 0:
				u_save = u[::8]
				us = np.concatenate((us,u_save.reshape((self.N,1))),axis=0)
		return us

	def generate_one_sol(self, p, test = False):
		u = self.u_approx(p)
		self.compile_output(u, p, test)

	def compile_output(self, u, p, test):
		if self.sampling_method == 0:
			U = u.reshape((self.N*self.Nt,1))
			if not test:		
				self.u_samples = np.hstack((self.u_samples,U)) if self.u_samples.size else U
			else:
				self.u_tests = np.hstack((self.u_tests,U)) if self.u_tests.size else U
		elif self.sampling_method == 1 or self.sampling_method == 2:
			if not test:
				X_f = self.create_tol_X(p)
				self.N0_samples = np.concatenate((self.N0_samples, X_f),axis = 0) if self.N0_samples.size else X_f
				self.u0_samples = np.concatenate((self.u0_samples,u),axis = 0) if self.u0_samples.size else u
			else:
				# u_tol = self.fill_BC(u, p)
				X_0 = self.create_tol_X(p)
				u_tol = u.reshape((self.N*self.Nt,1))
				self.N0_tests = np.concatenate((self.N0_tests, X_0),axis = 0) if self.N0_tests.size else X_0
				self.u0_tests = np.concatenate((self.u0_tests, u_tol),axis = 0) if self.u0_tests.size else u_tol

	def create_tol_X(self,p):
		X = self.X.reshape((self.N,1))
		T = self.T.reshape((self.Nt,1))
		Xs = np.ones((self.Nt,1))*X.reshape((1,self.N))
		Ts = T*np.ones((1,self.N))
		Xs = Xs.reshape((self.Nt*self.N,1))
		Ts = Ts.reshape((self.Nt*self.N,1))
		# P = np.log10(p)*np.ones((self.N,self.P_dim))
		P = p*np.ones((self.N*self.Nt,self.P_dim))
		X_f = np.concatenate((Xs,Ts,P),axis=1)
		return X_f

	def f(self, x):
		return 0.5*x**2

	def minmod(self, x, y):
		f = 0.5*(np.sign(x)+np.sign(y))*np.minimum(np.abs(x),np.abs(y))
		return f

	def vanleermc(self, x, y):
		f = self.minmod(0.5*(x+y),2*self.minmod(x,y))
		return f

	def generate_POD_samples(self):
		p_train, u_train = self.u_exact_train()
		if os.path.exists(self.path_env+"{2}_{1}_V_{0}.npy".format(self.L, self.N_p_train, self.name)):
			self.V = np.load(self.path_env+"{2}_{1}_V_{0}.npy".format(self.L, self.N_p_train, self.name))
		else:
			u,s,v = np.linalg.svd(u_train) 
			self.V = u[:,0:self.L]
			np.save(self.path_env+"{2}_{1}_V_{0}.npy".format(self.L, self.N_p_train, self.name),self.V)
		p_batch = p_train
		u_batch = u_train
		u_batch = self.V.T@u_batch
		xi_tf = tf.constant(p_batch,dtype = tf.float32)
		target_tf = tf.constant(u_batch.T,dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		x_tf = tf.constant((),shape = (self.N_p_train,0),dtype = tf.float32)
		N = tf.constant(self.N_p_train, dtype = tf.float32)
		self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Init'}
		samples_list = [self.X0_dict]
		return samples_list
	
	def generate_POD_tests(self):
		xi_tf = tf.constant(self.mu_mat_test.T ,dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		x_tf = tf.constant((),shape = (self.N_p_test,0),dtype = tf.float32)
		N = tf.constant(self.N_p_test, dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'N':N, 'type':'Init'}
		return X0_dict, self.u_tests

	def generate_PINN_samples(self):
		Ns = [self.Nf, self.Nb, self.Nn, self.N0]
		samples_list = []

		filename = "{1}_{0}.npz".format(Ns,self.name)
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

			if self.N0>0:
				self.X0 = npzfile['X0']
				self.u0 = npzfile['u0']
			else:
				self.X0_tf = None
				self.u0_tf = None
		else:
			np.random.seed(10)
			sampling_f = LHS(xlimits = self.x_p_domain)
			self.Xf = sampling_f(self.Nf)
			self.Xf[:,2] = np.power(10, self.Xf[:,2])
			target_f = np.zeros([self.Nf,1])

			sampling_b = LHS(xlimits = np.array([[0,1],[-3,-2]]))
			t_p_b = sampling_b(self.Nb//2)
			t_p_b[:,1] = np.power(10,t_p_b[:,1])
			self.Xb_d = np.concatenate((np.ones([self.Nb//2,1]),t_p_b),axis = 1)
			xb = np.concatenate((-np.ones([self.Nb//2,1]),t_p_b),axis = 1)
			self.Xb_d = np.concatenate((self.Xb_d,xb),axis = 0)
			self.ub_d = np.zeros((self.Nb,1))

			sampling_0 = LHS(xlimits = np.array([[-1,1],[-3,-2]]))
			x_p_0 = sampling_0(self.N0)
			x0 = x_p_0[:,[0]]
			p0 = x_p_0[:,[1]]
			p0 = np.power(10,p0)
			self.X0 = np.concatenate((x0,np.zeros((self.N0,1)),p0),axis = 1)
			self.u0 = -np.sin(np.pi*x0)

			np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, X0 = self.X0, u0 = self.u0)
		x_tf = tf.constant(self.Xf[:,[0]],dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
		t_tf = tf.constant(self.Xf[:,[1]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xf[:,[2]],dtype = tf.float32)
		target_tf = tf.constant(target_f, dtype = tf.float32)
		N = tf.constant(self.Nf, dtype = tf.float32)
		self.Xf_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Res'}
		samples_list.append(self.Xf_dict)

		x_tf = tf.constant(self.Xb_d[:,[0]],dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
		t_tf = tf.constant(self.Xb_d[:,[1]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xb_d[:,[2]],dtype = tf.float32)
		target_tf = tf.constant(self.ub_d, dtype = tf.float32)
		N = tf.constant(self.Nb, dtype = tf.float32)
		self.Xb_d_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_D'}
		samples_list.append(self.Xb_d_dict)

		x_tf = tf.constant(self.X0[:,[0]],dtype = tf.float32)
		y_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
		t_tf = tf.constant(self.X0[:,[1]],dtype = tf.float32)
		xi_tf = tf.constant(self.X0[:,[2]],dtype = tf.float32)
		target_tf = tf.constant(self.u0, dtype = tf.float32)
		N = tf.constant(self.N0, dtype = tf.float32)
		self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init"}
		samples_list.append(self.X0_dict)
		return samples_list

	def generate_PINN_tests(self):
		N = self.N0_tests.shape[0]
		x_tf = tf.constant(self.N0_tests[:,[0]],dtype = tf.float32)
		y_tf = tf.constant((),shape = (N,0),dtype = tf.float32)
		t_tf = tf.constant(self.N0_tests[:,[1]],dtype = tf.float32)
		xi_tf = tf.constant(self.N0_tests[:,[2]],dtype = tf.float32)
		target_tf = tf.constant(self.u0_tests, dtype = tf.float32)
		N = tf.constant(N, dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init"}
		return X0_dict, target_tf

	def test_NN(self, net, record_path = None):
		if record_path is not None:
			if os.path.exists(record_path):
				# rel_errs = np.loadtxt(record_path, delimiter="\n")
				# rel_errs = rel_errs.tolist()
				# if isinstance(rel_errs, float):
				# 	rel_errs = [rel_errs]
				pass
			else:
				with open(record_path, mode='w') as record:
					fields=['Net_struct','Net_setup','Sample','L','relative_err']
					record_writer = csv.writer(record, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
					record_writer.writerow(fields)
		X0_dict, u_test = self.u_exact_test()
		x_tf = X0_dict["x_tf"]
		y_tf = X0_dict["y_tf"]
		t_tf = X0_dict["t_tf"]
		xi_tf = X0_dict["xi_tf"]
		target_f = tf.zeros([self.Nt*self.N*self.N_p_test,1])
		u_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf)
		if self.sampling_method == 0:
			u_test_p = u_test_p.numpy()
			self.V = np.load(self.path_env+"{2}_{1}_V_{0}.npy".format(self.L, self.N_p_train, self.name))
			u_test_p = u_test_p@self.V.T
			u_test_p_grid = tf.constant(u_test_p, dtype = tf.float32)
			u_test_p_grid = tf.reshape(u_test_p_grid,(self.N_p_test,self.Nt,self.N))
			u_test_grid = tf.constant(u_test.T, dtype = tf.float32)
			u_test_grid = tf.reshape(u_test_grid,(self.N_p_test,self.Nt,self.N))
			f_res_grid = None
			N_record = [0, 0, 0, self.N_p_train]

		elif self.sampling_method == 1 or self.sampling_method == 2:
			N_record = [self.Nf, self.Nb, self.Nn, self.N0]
			f_res = net.compute_residual(x_tf, y_tf, t_tf, xi_tf, target_f)
			u_test_grid = tf.reshape(u_test,(self.N_p_test,self.Nt,self.N))
			u_test_p_grid = tf.reshape(u_test_p,(self.N_p_test,self.Nt,self.N))
			u_test_p_grid = tf.dtypes.cast(u_test_p_grid, tf.float32)
			f_res_grid = tf.reshape(f_res, (self.N_p_test,self.Nt,self.N))
			# inputs_range = self.select_region(N_test.numpy(),f_res_val,3,1e-3)

		err_grid = u_test_grid-u_test_p_grid
		err_test = tf.math.reduce_mean(tf.square(err_grid))

		relative_err_vec = tf.norm(err_grid,axis=[1,2])/tf.norm(u_test_grid,axis=[1,2])
		rel_err_test = tf.reduce_mean(relative_err_vec)
		if record_path is not None:
			list_info = [net.name, net.layers,N_record,self.L,rel_err_test.numpy()]
			with open(record_path, 'a') as f:
				writer = csv.writer(f)
				writer.writerow(list_info)
			# rel_errs.append(rel_err_test.numpy())
			# np.savetxt(record_path, rel_errs, delimiter =", ", fmt ='% s') 
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
			if self.sampling_method == 1 or self.sampling_method ==2:
				f_res_i = f_res_grid[i].numpy()

			# if figure_save_path is not None:
			# 	folder_path = "{1}xi_{0}".format(xi, figure_save_path)
			# 	if not os.path.exists(folder_path):
			# 		os.makedirs(folder_path)
			# if self.sampling_method == 1 or self.sampling_method ==2:	
			# 	scipy.io.savemat(folder_path+"/data.mat", {'true_solution':u_test_i, 'approximation': u_test_p_i, 'residual':f_res_i})
			# elif self.sampling_method == 0:
			# 	scipy.io.savemat(folder_path+"/data_POD.mat", {'true_solution':u_test_i, 'approximation': u_test_p_i})

			XX, TT = np.meshgrid(self.X,self.T)
			fig = plt.figure(1)
			ax = fig.gca(projection='3d')
			ax.plot_wireframe(XX, TT, u_test_p_i, color ="red")
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('u')
			ax.set_title('NN')
			fig1 = plt.figure(2)
			ax1 = fig1.gca(projection='3d')
			ax1.plot_wireframe(XX, TT, u_test_i)
			ax1.set_xlabel('x')
			ax1.set_ylabel('y')
			ax1.set_zlabel('u')
			ax1.set_title('FDM')
			plt.show()

	@tf.function
	def f_res(self, x_tf, y_tf, t_tf, xi_tf, u, u_x, u_y, u_t, u_xx, u_yy):
			f_u = u_t+u*u_x-xi_tf*u_xx
			return f_u

	@tf.function
	def neumann_bc(self, u_x, u_y):
		return
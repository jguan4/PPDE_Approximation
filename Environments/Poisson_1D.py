import numpy as np
from smt.sampling_methods import LHS
import scipy.io
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import csv

class Poisson_1D:
	def __init__(self, N_p_train, N_p_test, h, inner = False, sampling_method = 0, path_env = "./Environments/", L = 0):
		self.name = "Poisson"
		self.sampling_method = sampling_method
		self.u_dim = 1
		self.P_dim = 3
		self.inner = inner
		self.h = h
		self.domain = np.array([[-np.pi/2,np.pi/2]])
		self.plimits = plimits = np.array([[1.0, 3.0], [1.0, 3.0], [-0.5,0.5]])
		self.x_p_domain = np.array([[-np.pi/2,np.pi/2], [1.0, 3.0], [1.0, 3.0], [-0.5,0.5]])
		self.path_env = path_env

		if self.sampling_method == 0:
			self.lb = np.array([1.0, 1.0, -0.5])
			self.ub = np.array([3.0, 3.0, 0.5])
			self.state_space_size = self.P_dim
			self.output_space_size = None
			self.N_p_train = N_p_train[3]
			self.P_samples = np.array([])
			self.u_samples = np.array([])
			self.u_tests = np.array([])
			self.L = L
			self.generate_para()

		# return point-wise solutions, including parameters in input
		elif self.sampling_method == 1 or self.sampling_method == 2:
			self.state_space_size = self.u_dim+self.P_dim
			self.output_space_size  = 1
			self.lb = np.array([-np.pi/2, 1.0, 1.0, -0.5])
			self.ub = np.array([np.pi/2, 3.0, 3.0, 0.5])
			self.Nf = N_p_train[0]
			self.Nb = N_p_train[1]
			self.Nn = N_p_train[2]
			self.N0 = N_p_train[3]
			self.L = L

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
		self.N = int(1/h)+1
		self.nx = self.N-2
		self.x = np.linspace(self.domain[0,0],self.domain[0,1],num=self.N)
		self.x_in = self.x[1:-1]

	def generate_para(self):
		np.random.seed(10)
		sampling = LHS(xlimits=self.plimits)
		filename = self.path_env+"{1}_{0}.npy".format(self.N_p_train,self.name)
		# check if train parameters exist
		if os.path.exists(filename):
			self.mu_mat_train = np.load(filename)
		else: 
			self.mu_mat_train = sampling(self.N_p_train).T 
			np.save(filename,self.mu_mat_train)

	def generate_para_test(self):
		np.random.seed(10)
		sampling = LHS(xlimits=self.plimits)
		# check if test parameters exist
		if os.path.exists(self.path_env+"{1}_{0}.npy".format(self.N_p_test,self.name)):
			self.mu_mat_test = np.load(self.path_env+"{1}_{0}.npy".format(self.N_p_test,self.name))
		else:
			self.mu_mat_test = sampling(self.N_p_test).T
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

	def u_exact(self, p):
		mu1 = p[0]
		mu2 = p[1]
		mu3 = p[2]
		u = mu2*np.sin(2+mu1*self.x)*np.exp(mu3*self.x)
		return u

	def generate_A(self):
		pass

	def generate_F(self):
		pass

	def generate_one_sol(self, p, test = False):
		u = self.u_exact(p)
		self.compile_output(u, p, test)

	def compile_output(self, u, p, test):
		if self.sampling_method == 0:
			U = u.reshape((self.N,1))
			if not test:		
				self.u_samples = np.hstack((self.u_samples,U)) if self.u_samples.size else U
			else:
				self.u_tests = np.hstack((self.u_tests,U)) if self.u_tests.size else U
		elif self.sampling_method == 1 or self.sampling_method == 2:
			if not test:
				u = u.reshape((self.N,1))
				X_f = self.create_tol_X(p)
				self.N0_samples = np.concatenate((self.N0_samples, X_f),axis = 0) if self.N0_samples.size else X_f
				self.u0_samples = np.concatenate((self.u0_samples,u),axis = 0) if self.u0_samples.size else u
			else:
				X_0 = self.create_tol_X(p)
				u_tol = u.reshape((self.N,1))
				self.N0_tests = np.concatenate((self.N0_tests, X_0),axis = 0) if self.N0_tests.size else X_0
				self.u0_tests = np.concatenate((self.u0_tests, u_tol),axis = 0) if self.u0_tests.size else u_tol

	def create_tol_X(self,p):
		X = self.x.reshape((self.N,1))
		P = p*np.ones((self.N,self.P_dim))
		X_f = np.concatenate((X,P),axis=1)
		return X_f

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
			target_f = np.zeros([self.Nf,1])

			sampling_b = LHS(xlimits = self.plimits)
			x_p_b = sampling_b(self.Nb//2)
			lb = np.concatenate((-np.pi/2*np.ones([self.Nb//2,1]),x_p_b),axis = 1)
			ulb = x_p_b[:,[1]]*np.sin(2-x_p_b[:,[0]]*np.pi/2)*np.exp(-np.pi*x_p_b[:,[2]]/2)
			rb = np.concatenate((np.pi/2*np.ones([self.Nb//2,1]),x_p_b),axis = 1)
			urb = x_p_b[:,[1]]*np.sin(2+x_p_b[:,[0]]*np.pi/2)*np.exp(+np.pi*x_p_b[:,[2]]/2)

			self.Xb_d = np.concatenate((lb,rb),axis = 0)
			self.ub_d = np.concatenate((ulb,urb),axis = 0)

			if self.N0>0:
				sampling_0 = LHS(xlimits = self.x_p_domain)
				self.X0 = sampling_f(self.N0)
				self.u0 = self.u_exact(self.X0)
				np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, X0 = self.X0, u0 = self.u0)
			else:
				np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d)

		y_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
		x_tf = tf.constant(self.Xf[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xf[:,1::],dtype = tf.float32)
		target_tf = tf.constant(target_f, dtype = tf.float32)
		N = tf.constant(self.Nf, dtype = tf.float32)
		self.Xf_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Res'}
		samples_list.append(self.Xf_dict)

		y_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
		x_tf = tf.constant(self.Xb_d[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.Xb_d[:,1::],dtype = tf.float32)
		target_tf = tf.constant(self.ub_d, dtype = tf.float32)
		N = tf.constant(self.Nb, dtype = tf.float32)
		self.Xb_d_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_D'}
		samples_list.append(self.Xb_d_dict)

		if self.N0>0:
			y_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			t_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			x_tf = tf.constant(self.X0[:,[0]],dtype = tf.float32)
			xi_tf = tf.constant(self.X0[:,1::],dtype = tf.float32)
			target_tf = tf.constant(self.u0, dtype = tf.float32)
			N = tf.constant(self.N0, dtype = tf.float32)
			self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init"}
			samples_list.append(self.X0_dict)
		return samples_list

	def generate_PINN_tests(self):
		N = self.N0_tests.shape[0]
		y_tf = tf.constant((),shape = (N,0),dtype = tf.float32)
		t_tf = tf.constant((),shape = (N,0),dtype = tf.float32)
		x_tf = tf.constant(self.N0_tests[:,[0]],dtype = tf.float32)
		xi_tf = tf.constant(self.N0_tests[:,1::],dtype = tf.float32)
		target_tf = tf.constant(self.u0_tests, dtype = tf.float32)
		N = tf.constant(N, dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init"}
		return X0_dict, target_tf

	def test_NN(self, net, record_path = None):
		if record_path is not None:
			if os.path.exists(record_path):
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
		target_f = tf.zeros([self.N*self.N_p_test,1])
		u_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf)
		if self.sampling_method == 0:
			u_test_p = u_test_p.numpy()
			self.V = np.load(self.path_env+"{2}_{1}_V_{0}.npy".format(self.L, self.N_p_train, self.name))
			u_test_p = u_test_p@self.V.T
			u_test_p_grid = tf.constant(u_test_p, dtype = tf.float32)
			u_test_grid = tf.constant(u_test.T, dtype = tf.float32)
			f_res_grid = None
			N_record = [0, 0, 0, self.N_p_train]
		elif self.sampling_method == 1 or self.sampling_method == 2:
			N_record = [self.Nf, self.Nb, self.Nn, self.N0]
			f_res = net.compute_residual(x_tf, y_tf, t_tf, xi_tf, target_f)
			u_test_grid = tf.reshape(u_test,(self.N_p_test,self.N))
			u_test_p_grid = tf.reshape(u_test_p,(self.N_p_test,self.N))
			f_res_grid = tf.reshape(f_res, (self.N_p_test,self.N))
			
		err_grid = u_test_grid-u_test_p_grid
		err_test = tf.math.reduce_mean(tf.square(err_grid))

		relative_err_vec = tf.norm(err_grid,axis=1)/tf.norm(u_test_grid,axis=1)
		rel_err_test = tf.reduce_mean(relative_err_vec)
		if record_path is not None:
			list_info = [net.name, net.layers,N_record,self.L,rel_err_test.numpy()]
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
			if self.sampling_method == 1 or self.sampling_method ==2:
				f_res_i = f_res_grid[i].numpy()

			# if figure_save_path is not None:
			# 	folder_path = "{1}xi_{0}".format(xi, figure_save_path)
			# 	if not os.path.exists(folder_path):
			# 	    os.makedirs(folder_path)
			# scipy.io.savemat(folder_path+"/data.mat", {'true_solution':u_test_i, 'approximation': u_test_p_i, 'residual':f_res_i})

			fig, ax = plt.subplots()
			ax.plot(self.x, u_test_p_i, color ="red")
			ax.plot(self.x, u_test_i)
			ax.set_xlabel(r'$x$')
			ax.set_ylabel(r'$u$')
			plt.show()

	@tf.function
	def f_res(self, x_tf, y_tf, t_tf, xi_tf, u, u_x, u_y, u_t, u_xx, u_yy):
		xi1 = xi_tf[:,0:1]
		xi2 = xi_tf[:,1:2]
		xi3 = xi_tf[:,2:3]
		ux = xi2*tf.math.exp(xi3*x_tf)*(xi1*tf.math.cos(2+xi1*x_tf)+xi3*tf.math.sin(2+xi1*x_tf))
		uxx = xi2*tf.math.exp(xi3*x_tf)*((xi3**2-xi1**2)*tf.math.sin(2+xi1*x_tf)+2*xi2*xi3*tf.math.cos(2+xi1*x_tf))
		uu = xi2*tf.math.sin(2+xi1*x_tf)*tf.math.exp(xi3*x_tf)
		s = -tf.math.exp(uu)*(ux**2+uxx)
		f_u = -tf.math.exp(u)*(u_x**2+u_xx)-s
		return f_u

	@tf.function
	def neumann_bc(self, u_x, u_y):
		return

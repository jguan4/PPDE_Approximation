import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from smt.sampling_methods import LHS
import scipy.io
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

class CD_comp_steady:
	def __init__(self,N_p_train,N_p_test,h,inner = False, sampling_method = 0, path_env = "./Environments/", L = 0):
		# self.N_p = N_p
		self.name = "CD_comp_steady"
		self.sampling_method = sampling_method
		self.u_dim = 2
		self.P_dim = 3
		self.inner = inner
		self.h = h
		self.domain = np.array([[-1,1],[-1,1]])
		self.plimits = np.array([[-4, 0], [0, 1], [0, 1]])
		self.x_p_domain = np.array([[-1, 1], [-1, 1], [-4, 0], [0, 1], [0, 1]])
		self.path_env = path_env

		# return full solutions for parameters generated
		if self.sampling_method == 0:
			self.lb = np.array([1e-4, 0, 0])
			self.ub = np.array([1, 1, 1])
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
			self.lb = np.array([-1.0, -1.0, 1e-4, 0.0, 0.0])
			self.ub = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
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
		self.nx = self.N-2
		self.ny = self.N-1

		self.x = np.linspace(self.domain[0,0],self.domain[0,1],num=self.N)
		self.y = np.linspace(self.domain[1,0],self.domain[1,1],num=self.N)
		self.xt = self.x[1:-1]
		self.yt = self.y[1::]

		self.X, self.Y = np.meshgrid(self.x,self.y)
		self.X_in = self.X[1::,1:-1]
		self.Y_in = self.Y[1::,1:-1]

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
		L,F = self.generate_A_F(p)
		u = spsolve(L,F)
		return u

	def generate_A_F(self,p):
		xi0 = p[0]
		xi1 = p[1]
		xi2 = p[2]
		xi_tot = p[1]+p[2]
		h = 1/128
		N = int(2/h)+1
		nx = N-2
		ny = N-1

		x = np.linspace(self.domain[0,0],self.domain[0,1],num=N)
		y = np.linspace(self.domain[1,0],self.domain[1,1],num=N)
		xt = x[1:-1]
		yt = y[1::]
		X, Y = np.meshgrid(x,y)
		X_in = X[1::,1:-1]
		Y_in = Y[1::,1:-1]

		tot_dim = nx*ny

		I = sparse.eye(nx)
		Iy = sparse.eye(ny)
		e1 = np.ones((nx))
		e2 = e1.copy()
		e2[-1] = 2
		e3 = e1.copy()
		e3[-1] = 0
		T = sparse.diags([1,-4,1],[-1,0,1],shape=(nx,nx))
		S = sparse.diags([e2,e1],[-1,1],shape=(ny,ny))
		S1 = sparse.diags([-e3,e1],[-1,1],shape=(ny,ny))
		S2 = sparse.diags([-1,1],[-1,1],shape=(nx,nx))

		A = sparse.kron(Iy,T)+sparse.kron(S,I)
		B = sparse.kron(S1,I)
		C = sparse.kron(Iy,S2)

		Dy_diag = (xi1*(1+(X_in+1)**2/4)+xi2*(-2*X_in*(1-Y_in**2)))/xi_tot
		Dx_diag = xi2*(2*Y_in*(1-X_in**2))/xi_tot
		Dy = sparse.diags(Dy_diag.flatten(),0,shape = (tot_dim,tot_dim))
		Dx = sparse.diags(Dx_diag.flatten(),0,shape = (tot_dim,tot_dim))

		L = -xi0*A/(h**2)+Dy@B/(2*h)+Dx@C/(2*h)

		v1 = np.zeros((tot_dim,1))
		v2 = np.zeros((tot_dim,1))
		v3 = np.zeros((tot_dim,1))
		v1[0:nx] = xi1/xi_tot
		v2[range(0,tot_dim,nx),0] = xi1*(1 - ((1+yt)/2))**3/xi_tot
		v3[range(nx-1,tot_dim,nx),0] = (xi1*(1 - ((1+yt)/2))**2+xi2)/xi_tot

		F = xi0*(v1+v2+v3)/(h**2)+Dy@v1/(2*h)+Dx@(v2-v3)/(2*h)
		return L,F

	def generate_one_sol(self, p, test = False):
		u = self.u_approx(p)
		u = u.reshape((256, 255))
		self.compile_output(u, p, test)

	def compile_output(self, u, p, test):
		if self.sampling_method == 0:
			if self.inner:
				U = u.reshape((self.ny*self.nx,1))
			else:
				U = self.fill_BC(u,p)
				jump = int(self.h*128)
				U = U[::jump,::jump]
				U = U.reshape((self.N*self.N,1))
			if not test:		
				self.u_samples = np.hstack((self.u_samples,U)) if self.u_samples.size else U
			else:
				self.u_tests = np.hstack((self.u_tests,U)) if self.u_tests.size else U
		else:
			if not test:
				u_inner = u[0:-1,:]
				u_inner = u_inner.reshape(((self.ny-1)*self.nx,1))
				X_f = self.create_inner_X(p)

				self.N0_samples = np.concatenate((self.N0_samples, X_f),axis = 0) if self.N0_samples.size else X_f
				self.u0_samples = np.concatenate((self.u0_samples,u_inner),axis = 0) if self.u0_samples.size else u_inner
			else:
				u_tol = self.fill_BC(u,p)
				jump = int(self.h*128)
				u_tol = u_tol[::jump,::jump]
				u_tol = u_tol.reshape((self.N*self.N,1))
				X_0 = self.create_tol_X(p)
				self.N0_tests = np.concatenate((self.N0_tests, X_0),axis = 0) if self.N0_tests.size else X_0
				self.u0_tests = np.concatenate((self.u0_tests, u_tol),axis = 0) if self.u0_tests.size else u_tol
				
	def fill_BC(self,inner_u,p):
		xi1 = p[1]
		xi2 = p[2]
		xi_tot = xi1+xi2

		h = 1/128
		N = int(2/h)+1
		nx = N-2
		ny = N-1

		x = np.linspace(self.domain[0,0],self.domain[0,1],num=N)
		y = np.linspace(self.domain[1,0],self.domain[1,1],num=N)
		xt = x[1:-1]
		yt = y[1::]
		X, Y = np.meshgrid(x,y)
		X_in = X[1::,1:-1]
		Y_in = Y[1::,1:-1]

		u_temp = inner_u.reshape((ny,nx))
		U = np.zeros((N,N))
		U[1::,1:-1] = u_temp
		U[1::,0] = xi1*(1 - ((1+yt)/2))**3/xi_tot
		U[1::,-1] = (xi1*(1 - ((1+yt)/2))**2+xi2)/xi_tot
		U[0,:] = xi1/xi_tot
		return U

	def boundary_points(self,p):
		xi1 = p[1]
		xi2 = p[2]
		xi_tot = xi1+xi2
		lb = xi1*(1 - ((1+self.yt)/2))**3/xi_tot
		rb = (xi1*(1 - ((1+self.yt)/2))**2+xi2)/xi_tot
		ub = xi1/xi_tot*np.ones((self.N))
		db_n = np.zeros(self.xt.shape)

		BC_d = np.concatenate((lb.reshape((len(lb),1)),rb.reshape((len(rb),1)),ub.reshape((len(ub),1))),axis=0)
		BC_n = db_n.reshape((len(db_n),1))

		X_lb = np.concatenate((self.X[1::,[0]].T,self.Y[1::,[0]].T),axis=0)
		X_rb = np.concatenate((self.X[1::,[-1]].T,self.Y[1::,[-1]].T),axis=0)
		X_ub = np.concatenate((self.X[[0],:],self.Y[[0],:]),axis=0)
		X_db = np.concatenate((self.X[[-1],1:-1],self.Y[[-1],1:-1]),axis=0)
		
		X_d = np.concatenate((X_lb,X_rb,X_ub),axis = 1)
		X_n = X_db
		P_d = p*np.ones((X_d.shape[1],self.P_dim))
		P_n = p*np.ones((X_n.shape[1],self.P_dim))
		
		N_d = np.concatenate((X_d.T,P_d),axis=1)
		N_n = np.concatenate((X_n.T,P_n),axis=1)
		return N_d, BC_d, N_n, BC_n

	def create_inner_X(self, p):
		X_in = self.X_in[0:-1,:]
		Y_in = self.Y_in[0:-1,:]
		X = X_in.reshape(((self.ny-1)*self.nx,1))
		Y = Y_in.reshape(((self.ny-1)*self.nx,1))
		P = p*np.ones(((self.ny-1)*self.nx,self.P_dim))
		X_f = np.concatenate((X,Y,P),axis=1)
		return X_f

	def create_tol_X(self,p):
		X = self.X.reshape((self.N*self.N,1))
		Y = self.Y.reshape((self.N*self.N,1))
		P = p*np.ones((self.N*self.N,self.P_dim))
		X_f = np.concatenate((X,Y,P),axis=1)
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
			
			sampling_f = LHS(xlimits = self.x_p_domain)
			self.Xf = sampling_f(self.Nf)
			self.Xf[:,2] = np.power(10, self.Xf[:,2])

			sampling_b = LHS(xlimits = np.array([[-1, 1], [-4, 0], [0, 1], [0, 1]]))
			x_p_b = sampling_b(self.Nb//3)
			x_p_b[:,1] = np.power(10, x_p_b[:,1])
			xb = x_p_b[:,[0]]
			pb = x_p_b[:,1::]
			xi1 = pb[:,[1]]
			xi2 = pb[:,[2]]
			xi_tot = xi1+xi2
			lb = np.concatenate((-np.ones(xb.shape),xb, pb),axis = 1)
			ulb = xi1*(1 - ((1+xb)/2))**3/xi_tot
			rb = np.concatenate((np.ones(xb.shape),xb, pb),axis = 1)
			urb = (xi1*(1 - ((1+xb)/2))**2+xi2)/xi_tot
			tb = np.concatenate((xb, -np.ones(xb.shape), pb),axis = 1)
			utb = xi1/xi_tot
			self.Xb_d = np.concatenate((lb,rb,tb),axis = 0)
			self.ub_d = np.concatenate((ulb,urb,utb),axis = 0)

			x_p_n = sampling_b(self.Nn)
			x_p_n[:,1] = np.power(10, x_p_n[:,1])
			xn = x_p_n[:,[0]]
			pn = x_p_n[:,1::]
			self.Xb_n = np.concatenate((xn, np.ones(xn.shape), pn),axis = 1)
			self.ub_n = np.zeros(xn.shape)


			if self.N0>0:
				self.generate_para()
				for i in range(self.mu_mat_train.shape[1]):
					p = self.mu_mat_train[:,i]
					self.generate_one_sol(p)
				N0_tot = self.N0_samples.shape[0]
				N0_sel = np.random.choice(N0_tot, self.N0)
				self.X0 = self.N0_samples[N0_sel,:]
				self.X0_tf = tf.constant(self.X0, dtype = tf.float32)
				self.u0 = self.u0_samples[N0_sel,:]
				self.u0_tf = tf.constant(self.u0, dtype = tf.float32)
				np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, Xb_n = self.Xb_n, ub_n = self.ub_n, X0 = self.X0, u0 = self.u0)
			else:
				np.savez(self.path_env+"{0}".format(filename), Xf = self.Xf, Xb_d = self.Xb_d, ub_d = self.ub_d, Xb_n = self.Xb_n, ub_n = self.ub_n)

		x_tf = tf.constant(self.Xf[:,[0]],dtype = tf.float32)
		y_tf = tf.constant(self.Xf[:,[1]],dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.Nf,0),dtype = tf.float32)
		xi_tf = tf.constant(self.Xf[:,2::],dtype = tf.float32)
		target_tf = tf.constant(target_f, dtype = tf.float32)
		N = tf.constant(self.Nf, dtype = tf.float32)
		self.Xf_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'Res'}
		samples_list.append(self.Xf_dict)

		x_tf = tf.constant(self.Xb_d[:,[0]],dtype = tf.float32)
		y_tf = tf.constant(self.Xb_d[:,[1]],dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.Nb,0),dtype = tf.float32)
		xi_tf = tf.constant(self.Xb_d[:,2::],dtype = tf.float32)
		target_tf = tf.constant(self.ub_d, dtype = tf.float32)
		N = tf.constant(self.Nb, dtype = tf.float32)
		self.Xb_d_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_D'}
		samples_list.append(self.Xb_d_dict)

		x_tf = tf.constant(self.Xb_n[:,[0]],dtype = tf.float32)
		y_tf = tf.constant(self.Xb_n[:,[1]],dtype = tf.float32)
		t_tf = tf.constant((),shape = (self.Nn,0),dtype = tf.float32)
		xi_tf = tf.constant(self.Xb_n[:,2::],dtype = tf.float32)
		target_tf = tf.constant(self.ub_n, dtype = tf.float32)
		N = tf.constant(self.Nn, dtype = tf.float32)
		self.Xb_n_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':'B_N'}
		samples_list.append(self.Xb_n_dict)

		if self.N0>0:
			x_tf = tf.constant(self.X0[:,[0]],dtype = tf.float32)
			y_tf = tf.constant(self.X0[:,[1]],dtype = tf.float32)
			t_tf = tf.constant((),shape = (self.N0,0),dtype = tf.float32)
			xi_tf = tf.constant(self.X0[:,2::],dtype = tf.float32)
			target_tf = tf.constant(self.u0, dtype = tf.float32)
			N = tf.constant(self.N0, dtype = tf.float32)
			self.X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init"}
			samples_list.append(self.X0_dict)
		return samples_list

	def generate_PINN_tests(self):
		N = self.N0_tests.shape[0]
		x_tf = tf.constant(self.N0_tests[:,[0]],dtype = tf.float32)
		y_tf = tf.constant(self.N0_tests[:,[1]],dtype = tf.float32)
		t_tf = tf.constant((),shape = (N,0),dtype = tf.float32)
		xi_tf = tf.constant(self.N0_tests[:,2::],dtype = tf.float32)
		target_tf = tf.constant(self.u0_tests, dtype = tf.float32)
		N = tf.constant(N, dtype = tf.float32)
		X0_dict = {'x_tf':x_tf, 'y_tf':y_tf, 't_tf':t_tf, 'xi_tf':xi_tf, 'target':target_tf, 'N':N, 'type':"Init"}
		return X0_dict, target_tf

	def test_NN(self, net, record_path = None):
		if record_path is not None:
			if os.path.exists(record_path):
				rel_errs = np.loadtxt(record_path, delimiter="\n")
				rel_errs = rel_errs.tolist()
				if isinstance(rel_errs, float):
					rel_errs = [rel_errs]
			else:
				rel_errs = []
		X0_dict, u_test = self.u_exact_test()
		x_tf = X0_dict["x_tf"]
		y_tf = X0_dict["y_tf"]
		t_tf = X0_dict["t_tf"]
		xi_tf = X0_dict["xi_tf"]
		target_f = tf.zeros([self.N**2*self.N_p_test,1])
		u_test_p = net.forward(x_tf, y_tf, t_tf, xi_tf)
		if self.sampling_method == 0:
			u_test_p = u_test_p.numpy()
			self.V = np.load(self.path_env+"{2}_{1}_V_{0}.npy".format(self.L, self.N_p_train, self.name))
			u_test_p = u_test_p@self.V.T
			u_test_p_grid = tf.constant(u_test_p, dtype = tf.float32)
			u_test_p_grid = tf.reshape(u_test_p_grid,(self.N_p_test,self.N,self.N))
			u_test_grid = tf.constant(u_test.T, dtype = tf.float32)
			u_test_grid = tf.reshape(u_test_grid,(self.N_p_test,self.N,self.N))
			f_res_grid = None

		elif self.sampling_method == 1 or self.sampling_method == 2:
			f_res = net.compute_residual(x_tf, y_tf, t_tf, xi_tf, target_f)
			u_test_grid = tf.reshape(u_test,(self.N_p_test,self.N,self.N))
			u_test_p_grid = tf.reshape(u_test_p,(self.N_p_test,self.N,self.N))
			u_test_p_grid = tf.dtypes.cast(u_test_p_grid, tf.float32)
			f_res_grid = tf.reshape(f_res, (self.N_p_test,self.N,self.N))
			
		err_grid = u_test_grid-u_test_p_grid
			# inputs_range = self.select_region(N_test.numpy(),f_res_val,3,1e-3)

		err_test = tf.math.reduce_mean(tf.square(err_grid))

		relative_err_vec = tf.norm(err_grid,axis=[1,2])/tf.norm(u_test_grid,axis=[1,2])
		rel_err_test = tf.reduce_mean(relative_err_vec)
		if record_path is not None:
			rel_errs.append(rel_err_test.numpy())
			np.savetxt(record_path, rel_errs, delimiter =", ", fmt ='% s') 
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
			fig = plt.figure(1)
			ax = fig.gca(projection='3d')
			ax.plot_wireframe(self.X, self.Y, u_test_p_i, color ="red")
			ax.set_xlabel('x')
			ax.set_ylabel('y')
			ax.set_zlabel('u')
			ax.set_title('NN')
			fig1 = plt.figure(2)
			ax1 = fig1.gca(projection='3d')
			ax1.plot_wireframe(self.X, self.Y, u_test_i)
			ax1.set_xlabel('x')
			ax1.set_ylabel('y')
			ax1.set_zlabel('u')
			ax1.set_title('FDM')
			fig2 = plt.figure(3)
			ax2 = fig2.gca(projection='3d')
			ax2.plot_wireframe(self.X, self.Y, u_test_i-u_test_p_i)
			ax2.set_xlabel('x')
			ax2.set_ylabel('y')
			ax2.set_zlabel('u')
			ax2.set_title('Error')
			plt.show()
			# fig = plt.figure(figsize=plt.figaspect(0.5))
			# fig, ax = plt.subplots()
			# ax = fig.add_subplot(1, 2, 1)
			# ax = fig.add_subplot(1, 1, 1)
			# ax.plot(self.x, u_test_p_i, color ="red")
			# ax.plot(self.x, u_test_i)
			# ax.set_xlabel(r'$x$')
			# ax.set_ylabel(r'$u$')
			# fig.suptitle(r"$\xi$ = {0}".format(xi)))
			# if figure_save_path is not None:
				# plt.savefig("{1}/u_xi_{0}.png".format(xi,folder_path))
				# plt.cla()
				# plt.clf()
				# plt.close()
			# else:
				# plt.show()

			# fig1, ax1 = plt.subplots()
			# fig1 = plt.figure(2)
			# ax1 = fig.add_subplot(1, 2, 2)
			# ax1 = fig1.add_subplot(1, 1, 1)
			# ax1 = fig1.gca(projection='3d')
			# ax1.plot(self.x,f_res_i)
			# ax1.set_xlabel(r'$x$')
			# ax1.set_ylabel(r'$f(u,\xi)$')
			# fig1.suptitle(r"$\xi$ = {0}".format(xi))
			# if figure_save_path is not None:
				# plt.savefig("{1}/f_xi_{0}.png".format(xi,folder_path))
				# plt.close()
				# plt.cla()
				# plt.clf()
			# else:
				# plt.show()
			# ax1.semilogy(self.x, np.abs(u_test_p_i-u_test_i))
			# plt.show()


	@tf.function
	def f_res(self, x_tf, y_tf, t_tf, xi_tf, u, u_x, u_y, u_t, u_xx, u_yy):
		xi0 = xi_tf[:,0:1]
		xi1 = xi_tf[:,1:2]
		xi2 = xi_tf[:,2:3]
		xi_tol = xi1+xi2
		f_u = -xi0*(u_xx+u_yy)+(xi1/xi_tol)*(1+(x_tf+1)**2/4)*u_y+(xi2/xi_tol)*(2*y_tf*(1-x_tf**2)*u_x-2*x_tf*(1-y_tf**2)*u_y)
		return f_u

	@tf.function
	def neumann_bc(self, u_x, u_y):
		return u_y
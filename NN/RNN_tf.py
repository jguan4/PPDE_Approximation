import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
tf.random.set_seed(5)

class RNN_tf:
	def __init__(self, input_size, output_size, layers, env, Ntr, Ntr_t, Ntr_name, regular_alphas, type_weighting):
		self.env = env
		self.name = "RNN"
		self.input_size = input_size
		self.output_size = output_size
		self.x_dim = self.env.x_dim
		self.p_dim = self.env.p_dim
		self.x_disc = tf.constant(self.env.x_disc, dtype = tf.float32)
		self.h_init = tf.constant(self.env.h_init,dtype = tf.float32)
		self.layers = layers
		self.Ntr = Ntr
		self.Ntr_t = Ntr_t
		self.Ntr_name = Ntr_name
		self.regular_alphas = regular_alphas
		self.type_weighting = type_weighting
		self.loss_f_list = []
		self.loss_b_d_list = []
		self.initialize_NN()

	def initialize_NN(self):
		num_layers = len(self.layers) 
		self.weights = []
		self.weights_dims = []
		self.biases = []
		self.biases_dims = []
		self.weights_len = 0
		self.biases_len = 0

		l = 0
		W_name = "W{0}".format(l)
		W = self.xavier_init(size=[1,self.output_size*self.layers[0]], name = W_name)
		self.weights_dims.append((self.output_size,self.layers[0]))

		U_name = "U{0}".format(l)
		U = self.xavier_init(size=[1,self.layers[0]*self.x_dim], name = U_name)
		self.weights_dims.append((self.x_dim,self.layers[0]))
		
		Z_name = "Z{0}".format(l)
		Z = self.xavier_init(size=[1,self.layers[0]*self.p_dim], name = Z_name)
		self.weights_dims.append((self.p_dim,self.layers[0]))

		b_name = "bias{0}".format(l)
		# b1 = self.xavier_init(size = [1,self.layers[l]], name = b1_name)
		b = tf.Variable(tf.zeros(shape = [1, self.layers[0]]), name = b_name)
		self.biases_dims.append((1, self.layers[0]))
		
		self.weights_len += self.layers[0]*(self.output_size+self.x_dim+self.p_dim)
		self.biases_len += self.layers[0]

		self.weights.append(W)
		self.weights.append(U)
		self.weights.append(Z)
		self.biases.append(b)

		for l in range(1,num_layers-1):
			W_name = "W{0}".format(l)
			W = self.xavier_init(size=[1,self.layers[l]*self.layers[l+1]], name = W_name)

			b_name = "bias{0}".format(l)
			# b1 = self.xavier_init(size = [1,self.layers[l]], name = b1_name)
			b = tf.Variable(tf.zeros(shape = [1, self.layers[l+1]]), name = b_name)
			
			self.weights_dims.append((self.layers[l], self.layers[l+1]))
			self.biases_dims.append((1, self.layers[l+1]))

			self.weights_len += self.layers[l]*self.layers[l+1]
			self.biases_len += self.layers[l+1]

			self.weights.append(W)
			self.biases.append(b)

		W_name = "weight_{0}".format("ouput")
		b_name = "biase_{0}".format("ouput")
		W = self.xavier_init(size=[1, self.layers[-1]*self.output_size], name = W_name)
		# b = self.xavier_init(size = [1, self.output_size], name = b_name)
		b =tf.Variable(tf.zeros(shape = [1, self.output_size]), name = b_name)

		self.weights_len += self.output_size*self.layers[-1]
		self.biases_len += self.output_size	

		self.weights_dims.append((self.layers[-1], self.output_size))
		self.biases_dims.append((1, self.output_size))

		self.weights.append(W)
		self.biases.append(b)


	def xavier_init(self, size, name):
		xavier_stddev = np.sqrt(6)/np.sqrt(np.sum(size)) #np.sqrt(2/(in_dim + out_dim))
		return tf.Variable(tf.random.uniform(size, minval = -np.sqrt(6)/np.sqrt(np.sum(size)), maxval = np.sqrt(6)/np.sqrt(np.sum(size))), dtype=tf.float32, name = name)

	@tf.function
	def forward(self, x_tf, y_tf, t_tf, xi_tf):
		num_layers = len(self.layers)  
		X = tf.concat((x_tf, y_tf, t_tf, xi_tf), axis = -1)
		H = self.h_init
		h = self.h_init
		for i in range(len(self.x_disc)):
			x = self.x_disc[i]
			W = tf.reshape(self.weights[0], self.weights_dims[0])
			U = tf.reshape(self.weights[1], self.weights_dims[1])
			Z = tf.reshape(self.weights[2], self.weights_dims[2])
			b = tf.reshape(self.biases[0], self.biases_dims[0])
			h = tf.keras.activations.tanh(tf.matmul(h,W)+x*U+tf.matmul(X,Z)+b)
			for l in range(1,num_layers-1):
				W = tf.reshape(self.weights[l+2], self.weights_dims[l+2])
				b = self.biases[l]
				h = tf.keras.activations.tanh(tf.matmul(h, W) + b)	
			W = tf.reshape(self.weights[-1], self.weights_dims[-1])
			b = tf.reshape(self.biases[-1], self.biases_dims[-1])
			h = tf.matmul(h,W)+b
			H = tf.concat((H,h),axis = 1)
		return H

	@tf.function
	def derivatives(self, x_tf, y_tf, t_tf, xi_tf):
		with tf.GradientTape(persistent = True) as tape1:
			tape1.watch(x_tf)
			tape1.watch(y_tf)
			tape1.watch(t_tf)
			with tf.GradientTape(persistent = True) as tape:
				tape.watch(x_tf)
				tape.watch(y_tf)
				tape.watch(t_tf)
				H = self.forward(x_tf, y_tf, t_tf, xi_tf)
			u_x = tape.gradient(H, x_tf)
			u_y = tape.gradient(H, y_tf)
			u_t = tape.gradient(H, t_tf)
		u_xx = tape1.gradient(u_x, x_tf)
		u_yy = tape1.gradient(u_y, y_tf)
		return H, u_x, u_y, u_t, u_xx, u_yy

	@tf.function
	def compute_residual(self, x_tf, y_tf, t_tf, xi_tf, target):
		u, u_x, u_y, u_t, u_xx, u_yy = self.derivatives(x_tf, y_tf, t_tf, xi_tf)
		f_res = self.env.f_res(x_tf, y_tf, t_tf, xi_tf, u, u_x, u_y, u_t, u_xx, u_yy)
		f_err = f_res - target
		return f_err

	@tf.function 
	def compute_neumann(self, x_tf, y_tf, t_tf, xi_tf, target):
		u, u_x, u_y, u_t, u_xx, u_yy = self.derivatives(x_tf, y_tf, t_tf, xi_tf)
		ub_n_p = self.env.neumann_bc(u_x, u_y)
		err = ub_n_p - target
		return err

	@tf.function
	def compute_solution(self, x_tf, y_tf, t_tf, xi_tf, target):
		u_p = self.forward(x_tf, y_tf, t_tf, xi_tf)
		err = u_p - target
		return err

	# @tf.function
	def loss(self, samples_list, save_toggle = False):
		loss_val = tf.constant(0, dtype = tf.float32)
		for i in range(len(samples_list)):
			dict_i = samples_list[i]
			name_i = dict_i["type"]
			x_tf = dict_i["x_tf"]
			y_tf = dict_i["y_tf"]
			t_tf = dict_i["t_tf"]
			xi_tf = dict_i["xi_tf"]
			target = dict_i["target"]
			N = dict_i["N"]

			if name_i == "Res":
				f_res = self.compute_residual(x_tf, y_tf, t_tf, xi_tf, target)
				f_u = f_res*np.sqrt(self.type_weighting[0]/N)
				loss_f = tf.math.reduce_sum(f_u**2)/2
				loss_val = loss_val + loss_f

			elif name_i == "B_D":
				err_do = self.compute_solution(x_tf, y_tf, t_tf, xi_tf, target)
				err_d = err_do*np.sqrt(self.type_weighting[1]/(N))
				loss_d = tf.math.reduce_sum(err_d**2)/2
				loss_val = loss_val + loss_d
	
			elif name_i == "B_N":
				err_n = self.compute_neumann(x_tf, y_tf, t_tf, xi_tf,target)
				err_n = (err_n)*np.sqrt(self.type_weighting[2]/(N))
				loss_n = tf.math.reduce_sum(err_n**2)/2
				loss_val = loss_val + loss_n

			elif name_i == "Init":
				err_0 = self.compute_solution(x_tf, y_tf, t_tf, xi_tf, target)
				err_0 = (err_0)*np.sqrt(self.type_weighting[3]/(N))
				loss_0 = tf.math.reduce_sum(err_0**2)/2
				loss_val = loss_val + loss_0

		return loss_val

	@tf.function
	def construct_Jacobian_solution(self, x_tf, y_tf, t_tf, xi_tf, target, N):
		with tf.GradientTape(persistent = True) as tape:
			err = self.compute_solution(x_tf, y_tf, t_tf, xi_tf, target)*tf.math.sqrt(self.type_weighting[1]/(N))
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)

		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	@tf.function
	def construct_Jacobian_residual(self, x_tf, y_tf, t_tf, xi_tf, target, N):
		with tf.GradientTape(persistent = True) as tape:
			f_res = self.compute_residual(x_tf, y_tf, t_tf, xi_tf, target)
			err = (f_res)*tf.math.sqrt(self.type_weighting[0]/(N))
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)
		if biases_jacobians[-1] is None:
			biases_jacobians[-1] = tf.zeros([tf.shape(biases_jacobians[0])[0], self.output_size, 1, self.output_size])
		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	@tf.function
	def construct_Jacobian_neumann(self, x_tf, y_tf, t_tf, xi_tf, target, N):
		with tf.GradientTape(persistent = True) as tape:
			err = self.compute_neumann(x_tf, y_tf, t_tf, xi_tf, target)*tf.math.sqrt(self.type_weighting[2]/(N))
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)
		if biases_jacobians[-1] is None:
			biases_jacobians[-1] = tf.zeros([tf.shape(biases_jacobians[0])[0], self.output_size, 1, self.output_size])
		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	def construct_Gradient(self, samples_list):
		with tf.GradientTape(persistent = True) as tape:
			loss_val = self.loss(samples_list)
		weights_grads = tape.gradient(loss_val, self.weights)
		biases_grads = tape.gradient(loss_val, self.biases)
		if biases_grads[-1] is None:
			biases_grads[-1] = tf.zeros([1, self.output_size])
		grads_tol_W = tf.transpose(tf.concat(weights_grads, axis = -1))
		grads_tol_b = tf.transpose(tf.concat(biases_grads, axis = -1))
		return loss_val, grads_tol_W, grads_tol_b

	def construct_Hessian(self, loss_val, tape_loss):		
		num_layers = len(self.layers)  

		with tape_loss:
			loss_grad_W, loss_grad_b = self.construct_Gradient(loss_val, tape_loss)
		hessian_W = tape_loss.jacobian(loss_grad_W, self.weights)
		hessian_b = tape_loss.jacobian(loss_grad_b, self.biases)
		hessian_W = tf.squeeze(tf.concat(hessian_W, axis = 3))
		if hessian_b[-1] == None:
			hessian_b[-1] = tf.zeros([self.biases_len, self.output_size, 1, self.output_size])
		hessian_b = tf.squeeze(tf.concat(hessian_b, axis = 3))
		return hessian_W, hessian_b

	def update_weights_biases(self, weights_update, biases_update):
		num_layers = len(self.biases) 
		W_ind_count = 0
		b_ind_count = 0 
		
		l = 0
		W_len = np.prod(self.weights_dims[3*l])
		U_len = np.prod(self.weights_dims[3*l+1])
		Z_len = np.prod(self.weights_dims[3*l+2])
		b_len = np.prod(self.biases_dims[l])
		W_update = weights_update[W_ind_count:W_ind_count+W_len]
		W_ind_count += W_len
		U_update = weights_update[W_ind_count:W_ind_count+U_len]
		W_ind_count += U_len
		Z_update = weights_update[W_ind_count:W_ind_count+Z_len]
		W_ind_count += Z_len

		b_update = biases_update[b_ind_count:b_ind_count+b_len]
		b_ind_count += b_len
		self.weights[3*l].assign_add(tf.reshape(W_update, [1, W_len]))
		self.weights[3*l+1].assign_add(tf.reshape(U_update, [1, U_len]))
		self.weights[3*l+2].assign_add(tf.reshape(Z_update, [1, Z_len]))
		self.biases[l].assign_add(tf.reshape(b_update, [1, b_len]))
		for l in range(1,num_layers):
			W_len = np.prod(self.weights_dims[l+2])
			b_len = np.prod(self.biases_dims[l])
			W_update = weights_update[W_ind_count:W_ind_count+W_len]
			b_update = biases_update[b_ind_count:b_ind_count+b_len]
			self.weights[l+2].assign_add(tf.reshape(W_update, [1, W_len]))
			self.biases[l].assign_add(tf.reshape(b_update, [1, b_len]))

			W_ind_count += W_len
			b_ind_count += b_len

	def set_weights_biases(self, new_weights, new_biases, lin = False):
		num_layers = len(self.biases) 
		if not lin:
			l=0
			self.weights[3*l].assign(new_weights[3*l])
			self.weights[3*l+1].assign(new_weights[3*l+1])
			self.weights[3*l+2].assign(new_weights[3*l+2])
			self.biases[l].assign(new_biases[l])
			for l in range(1,num_layers):
				self.weights[l+2].assign(new_weights[l+2])
				self.biases[l].assign(new_biases[l])
		if lin:
			W_ind_count = 0
			b_ind_count = 0 
			l = 0
			W_len = np.prod(self.weights_dims[3*l])
			U_len = np.prod(self.weights_dims[3*l+1])
			Z_len = np.prod(self.weights_dims[3*l+2])
			b_len = np.prod(self.biases_dims[l])
			W_new = new_weights[W_ind_count:W_ind_count+W_len]
			W_ind_count += W_len
			U_new = new_weights[W_ind_count:W_ind_count+U_len]
			W_ind_count += U_len
			Z_new = new_weights[W_ind_count:W_ind_count+Z_len]
			W_ind_count += Z_len
			b_new = new_biases[b_ind_count:b_ind_count+b_len]
			b_ind_count += b_len
			self.weights[3*l].assign(tf.reshape(W_new, [1, W_len]))
			self.weights[3*l+1].assign(tf.reshape(U_new, [1, U_len]))
			self.weights[3*l+2].assign(tf.reshape(Z_new, [1, Z_len]))
			self.biases[l].assign(tf.reshape(b_new, [1, b_len]))
			for l in range(1,num_layers):
				W_len = np.prod(self.weights_dims[l+2])
				b_len = np.prod(self.biases_dims[l])
				W_new = new_weights[W_ind_count:W_ind_count+W_len]
				b_new = new_biases[b_ind_count:b_ind_count+b_len]
				W_new = tf.reshape(W_new, [1, W_len])
				b_new = tf.reshape(b_new, [1, b_len])
				self.weights[l+2].assign(W_new)
				self.biases[l].assign(b_new)
				W_ind_count += W_len
				b_ind_count += b_len
		return self.weights, self.biases

	def get_weights_biases(self):
		weights = []
		biases = []
		num_layers = len(self.biases)  
		l = 0
		W_len = np.prod(self.weights_dims[3*l])
		U_len = np.prod(self.weights_dims[3*l+1])
		Z_len = np.prod(self.weights_dims[3*l+2])
		b_len = np.prod(self.biases_dims[l])
		W = self.weights[3*l]
		U = self.weights[3*l+1]
		Z = self.weights[3*l+2]
		b = self.biases[l]
		W_copy = tf.identity(W)
		U_copy = tf.identity(U)
		Z_copy = tf.identity(Z)
		b_copy = tf.identity(b)
		weights.append(W_copy)
		weights.append(U_copy)
		weights.append(Z_copy)
		biases.append(b_copy)
		weights_lin = tf.reshape(W_copy, [1, W_len])
		weights_lin = tf.concat((weights_lin,tf.reshape(U_copy, [1, U_len])),axis = 1)
		weights_lin = tf.concat((weights_lin,tf.reshape(Z_copy, [1, Z_len])),axis = 1)
		biases_lin = tf.reshape(b_copy, [1, b_len])
		for l in range(1,num_layers):
			W = self.weights[l+2]
			b = self.biases[l]
			W_len = np.prod(self.weights_dims[l+2])
			b_len = np.prod(self.biases_dims[l])
			W_copy = tf.identity(W)
			b_copy = tf.identity(b)
			weights.append(W_copy)
			biases.append(b_copy)
			weights_lin = tf.concat((weights_lin,tf.reshape(W_copy, [1, W_len])),axis = 1)
			biases_lin = tf.concat((biases_lin,tf.reshape(b_copy, [1, b_len])),axis = 1)
		return weights, biases, weights_lin, biases_lin

	def set_weights_loss(self, samples_list, new_weights, new_biases):
		self.set_weights_biases(tf.squeeze(new_weights),tf.squeeze(new_biases),lin=True)
		loss_val_tf, _, _, _ = self.loss(samples_list, self.Ntr)
		return loss_val_tf.numpy()

	def line_search(self, f, x, y, delta_x, delta_y ,tau, alpha = 0.01):
		incre_x = tau*delta_x
		incre_y = tau*delta_y
		x_new = x + incre_x
		y_new = y + incre_y
		rhs = f(x, y) - alpha*np.sum([np.sum(incre_x*delta_x),np.sum(incre_y*delta_y)])
		while f(x_new, y_new)>rhs:
			tau = 0.5*tau
			incre_x = tau*delta_x
			incre_y = tau*delta_y
			x_new = x + incre_x
			y_new = y + incre_y
			rhs = f(x, y) - alpha*np.sum([np.sum(incre_x*delta_x),np.sum(incre_y*delta_y)])
		return tau, x_new, y_new

	def select_batch(self, samples_list, batch_size = 64): 
		sample_batch_list = []
		for i in range(len(samples_list)):
			sample_i = samples_list[i]
			x_i = sample_i["x_tf"]
			y_i = sample_i["y_tf"]
			t_i = sample_i["t_tf"]
			xi_i = sample_i["xi_tf"]
			output_i = sample_i["target"]
			type_str = sample_i["type"]
			N = sample_i["N"]

			if N>batch_size:
				N_sel = np.random.choice(N,batch_size).reshape([batch_size,1])
				target = tf.Variable(tf.gather_nd(output_i,N_sel), trainable = False, name = "batch_output")
				x_batch = tf.Variable(tf.gather_nd(x_i,N_sel), trainable = False, name = "batch_x")
				y_batch = tf.Variable(tf.gather_nd(y_i,N_sel), trainable = False, name = "batch_y")
				t_batch = tf.Variable(tf.gather_nd(t_i,N_sel), trainable = False, name = "batch_t")
				xi_batch = tf.Variable(tf.gather_nd(xi_i,N_sel), trainable = False, name = "batch_xi")

				N = tf.constant(batch_size, dtype = tf.float32)
				sample_dict = {'x_tf':x_batch, 'y_tf':y_batch, 't_tf':t_batch, 'xi_tf':xi_batch, 'target':target, 'N':N, 'type':type_str}
				sample_batch_list.append(sample_dict)

				samples_i_dict = {"type":Ntr_name_i, "output":us_batch, "input":Xs_batch_list, "N":N}
				sample_batch_list.append(samples_i_dict)
			else:
				sample_batch_list.append(sample_i)
		return sample_batch_list

	def save_weights_biases(self, filename):
		_, _, weights_lin, biases_lin = self.get_weights_biases()
		np.savez(filename, weights_lin = weights_lin.numpy(), biases_lin = biases_lin.numpy())

	def load_weights_biases(self, filename):
		npzfile = np.load(filename)
		weights_lin = npzfile['weights_lin']
		biases_lin = npzfile['biases_lin']
		self.set_weights_biases(np.squeeze(weights_lin),np.squeeze(biases_lin),lin = True)
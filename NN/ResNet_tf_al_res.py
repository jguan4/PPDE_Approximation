import numpy as np
import sys
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
tf.random.set_seed(5)

class ResNet_tf_al_res:
	def __init__(self, input_size, output_size, layers, env, regular_alphas):
		self.env = env
		self.name = "ResNet"
		self.input_size = input_size
		self.output_size = output_size
		self.layers = layers
		self.regular_alphas = tf.constant(regular_alphas[0], dtype = tf.float32)
		self.lb = tf.constant(self.env.lb, dtype = tf.float32)
		self.ub = tf.constant(self.env.ub, dtype = tf.float32)
		self.initialize_NN()
		self.loss_f_list = []
		self.loss_b_d_list = []
		self.loss_b_n_list = []
		self.loss_init_list = []
		
	def load_mat(self):
		self.A = np.load('/home/jj/Documents/Git/PPDE_Approximation/Environments/NS_AL4/NS_AL4_[  0   0   0 400]_3_ver2_Fnst.npy')
		self.A = self.A.astype(np.float32)


	def initialize_NN(self):
		# initializing lists to store weights and dimension of weights
		num_layers = len(self.layers) 
		self.weights = []
		self.weights_dims = []
		self.biases = []
		self.biases_dims = []
		self.weights_len = 0
		self.biases_len = 0

		# create input layer
		W_name = "weight_{0}".format("input")
		W = self.xavier_init(size=[1, self.input_size*self.layers[0]], name = W_name)
		self.weights_len += self.input_size*self.layers[0]
		b_name = "bias_{0}".format("input")
		b = tf.Variable(tf.zeros(shape = [1, self.layers[0]]), name = b_name)
		self.biases_len += self.layers[0]

		self.weights_dims.append((self.input_size, self.layers[0]))
		self.biases_dims.append((1, self.layers[0]))
		self.weights.append(W)
		self.biases.append(b)

		# create hidden layers
		for l in range(num_layers-1):
			W_name = "weight_{0}".format(l)
			W = self.xavier_init(size=[1, self.layers[l]*self.layers[l+1]], name = W_name)
			self.weights_len += self.layers[l]*self.layers[l+1]

			b_name = "bias_{0}".format(l)
			b = tf.Variable(tf.zeros(shape = [1, self.layers[l+1]]), name = b_name)
			self.biases_len += self.layers[l+1]

			self.weights_dims.append((self.layers[l], self.layers[l+1]))
			self.biases_dims.append((1, self.layers[l+1]))

			self.weights.append(W)
			self.biases.append(b)

		# create output layer
		W_name = "weight_{0}".format("ouput")
		b_name = "biase_{0}".format("ouput")
		W = self.xavier_init(size=[1, self.layers[-1]*self.output_size], name = W_name)
		b =tf.Variable(tf.zeros(shape = [1, self.output_size]), name = b_name)

		self.weights_len += self.output_size*self.layers[-1]
		self.biases_len += self.output_size	

		self.weights_dims.append((self.layers[-1], self.output_size))
		self.biases_dims.append((1, self.output_size))

		self.weights.append(W)
		self.biases.append(b)

	def xavier_init(self, size, name):
		xavier_stddev = np.sqrt(6)/np.sqrt(np.sum(size)) 
		return tf.Variable(tf.random.normal(size, mean = 0,\
		 stddev = xavier_stddev), dtype=tf.float32, name = name)

	@tf.function
	def forward(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf):
		num_layers = len(self.weights) 
		X = tf.concat((x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf), axis = -1)
		H = X
		W = tf.reshape(self.weights[0], self.weights_dims[0])
		b = self.biases[0]
		H = tf.keras.activations.tanh(tf.matmul(H, W) + b)
		# H = (X - self.lb)/(self.ub - self.lb)
		# for l in range(num_layers-1):
		for l in range(1,num_layers-1):
			W = tf.reshape(self.weights[l], self.weights_dims[l])
			b = self.biases[l]
			# H = tf.keras.activations.tanh(tf.matmul(H, W) + b)	
			H = 0.5*H+tf.keras.activations.tanh(tf.matmul(H, W) + b)	
		W = tf.reshape(self.weights[-1], self.weights_dims[-1])
		b = self.biases[-1]
		H = tf.matmul(H, W) + b
		# H1,H2 = tf.split(H,2,axis=1)
		return H

	@tf.function
	def derivatives(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf):
		with tf.GradientTape(persistent = True) as tape1:
			tape1.watch(x_tf)
			tape1.watch(y_tf)
			tape1.watch(t_tf)
			with tf.GradientTape(persistent = True) as tape:
				tape.watch(x_tf)
				tape.watch(y_tf)
				tape.watch(t_tf)
				H1,H2 = self.forward(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf)
			u1_x = tape.gradient(H1, x_tf)
			u1_y = tape.gradient(H1, y_tf)
			u1_t = tape.gradient(H1, t_tf)

			u2_x = tape.gradient(H2, x_tf)
			u2_y = tape.gradient(H2, y_tf)
			u2_t = tape.gradient(H2, t_tf)

		u1_xx = tape1.gradient(u1_x, x_tf)
		u1_xy = tape1.gradient(u1_x, y_tf)
		u1_yy = tape1.gradient(u1_y, y_tf)
		u2_xx = tape1.gradient(u2_x, x_tf)
		u2_xy = tape1.gradient(u2_x, y_tf)
		u2_yy = tape1.gradient(u2_y, y_tf)
		return H1, H2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy

	@tf.function
	def compute_residual(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target):
		u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy = self.derivatives(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf)
		f_res1,f_res2 = self.env.f_res(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy)
		target1,target2 =tf.split(target, 2, axis = 1)
		f_err1 = f_res1 - target1
		f_err2 = f_res2 - target2
		f_err = tf.concat((f_err1, f_err2),axis=0)
		return f_err

	@tf.function
	def compute_div(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target):
		u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy = self.derivatives(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf)
		div = u1_x+u2_y 
		div_err = div-target
		return div_err

	@tf.function 
	def compute_neumann(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target):
		u1, u2, u1_x, u1_y, u1_t, u1_xx, u1_yy, u1_xy, u2_x, u2_y, u2_t, u2_xx, u2_yy, u2_xy = self.derivatives(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf)
		ub_n_p1, ub_n_p2  = self.env.neumann_bc(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, u1_x, u1_y, u2_x, u2_y)
		target1,target2 =tf.split(target, 2, axis = 1)
		err1 = ub_n_p1 - target1
		err2 = ub_n_p2 - target2
		err = tf.concat((err1, err2),axis=0)
		return err

	# @tf.function
	def compute_solution(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target):
		u = self.forward(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf)
		# err = u-target
		err = []
		for i in range(u.shape[0]):
			Fnst = self.A[i,:,:]
			rhs = tf.reshape(tf.concat((t1_tf[i,:], t2_tf[i,:]),axis=0),(u.shape[1],1))
			erri = tf.linalg.matmul(Fnst,tf.reshape(u[i,:],[u.shape[1],1]))-rhs
			err.append(tf.reshape(erri,[u.shape[1]]))
		err = tf.stack(err)

		# target1,target2 =tf.split(target, 2, axis = 1)
		# err1 = u1_p - target1
		# err2 = u2_p - target2
		# err = tf.concat((err1, err2),axis=0)
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
			w1_tf = dict_i["w1_tf"]
			w2_tf = dict_i["w2_tf"]
			t1_tf = dict_i["t1_tf"]
			t2_tf = dict_i["t2_tf"]
			target = dict_i["target"]
			N = dict_i["N"]
			weight = dict_i["weight"]

			if name_i == "Res":
				f_res = self.compute_residual(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)
				f_u = f_res*np.sqrt(weight/N)
				loss_f = tf.math.reduce_sum(f_u**2)/2
				loss_val = loss_val + loss_f
				if save_toggle:
					# pass
					self.loss_f_list.append(loss_f.numpy())

			elif name_i == "B_D":
				err_do = self.compute_solution(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)
				err_d = err_do*np.sqrt(weight/N)
				loss_d = tf.math.reduce_sum(err_d**2)/2
				loss_val = loss_val + loss_d
				if save_toggle:
					# pass
					self.loss_b_d_list.append(loss_d.numpy())
					
			elif name_i == "B_N":
				err_n = self.compute_neumann(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf,target)
				err_n = (err_n)*np.sqrt(weight/N)
				loss_n = tf.math.reduce_sum(err_n**2)/2
				loss_val = loss_val + loss_n
				if save_toggle:
					# pass
					self.loss_b_n_list.append(loss_n.numpy())

			elif name_i == "Init":
				loss_0 = 0.0
				err_0 = self.compute_solution(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)
				# for i in range(err_0.shape[0]):
					# err_0i = tf.reshape(err_0[i],[err_0.shape[1],1])
					# print(err_0i)
					# input()
					#<L^{-1}e,e>
					# loss_0i = tf.math.reduce_sum(tf.linalg.solve(self.A, err_0i)*err_0i)
					# loss_0 = loss_0 + loss_0i*(weight/(2*N))
					#<L^{-1}r,r>
				err_0 = (err_0)*np.sqrt(weight/N)
				loss_0 = tf.math.reduce_sum(err_0**2)/2
				loss_val = loss_val + loss_0
				if save_toggle:
					self.loss_init_list.append(loss_0.numpy())

			elif name_i == "Div":
				f_res = self.compute_div(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)
				f_u = f_res*np.sqrt(weight/N)
				loss_f = tf.math.reduce_sum(f_u**2)/2
				loss_val = loss_val + loss_f

		if self.regular_alphas != 0:
			weights = tf.concat(self.weights, axis = -1)
			biases = tf.concat(self.biases, axis = -1)
			loss_val = loss_val + self.regular_alphas*(tf.norm(weights)**2+tf.norm(biases)**2)
		return loss_val

	@tf.function
	def construct_Jacobian_solution(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target, N, weight):
		with tf.GradientTape(persistent = True) as tape:
			err = self.compute_solution(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)*tf.math.sqrt(weight/N)
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)

		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	@tf.function
	def construct_Jacobian_residual(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target, N, weight):
		with tf.GradientTape(persistent = True) as tape:
			f_res = self.compute_residual(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)
			err = (f_res)*tf.math.sqrt(weight/N)
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)
		if biases_jacobians[-1] is None:
			biases_jacobians[-1] = tf.zeros([tf.shape(biases_jacobians[0])[0], \
				1, 1, self.output_size])
		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	@tf.function
	def construct_Jacobian_reduced_residual(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target, N, weight):
		with tf.GradientTape(persistent = True) as tape:
			f_res = self.compute_reduced_residual(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)
			err = (f_res)*tf.math.sqrt(weight/N)
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)
		if biases_jacobians[-1] is None:
			biases_jacobians[-1] = tf.zeros([tf.shape(biases_jacobians[0])[0], \
				self.output_size, 1, self.output_size])
		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	@tf.function
	def construct_Jacobian_div(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target, N, weight):
		with tf.GradientTape(persistent = True) as tape:
			f_res = self.compute_div(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)
			err = (f_res)*tf.math.sqrt(weight/N)
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)
		if biases_jacobians[-1] is None:
			biases_jacobians[-1] = tf.zeros([tf.shape(biases_jacobians[0])[0], \
				self.output_size, 1, self.output_size])
		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	@tf.function
	def construct_Jacobian_neumann(self, x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target, N, weight):
		with tf.GradientTape(persistent = True) as tape:
			err = self.compute_neumann(x_tf, y_tf, t_tf, xi_tf, w1_tf, w2_tf, t1_tf, t2_tf, target)*tf.math.sqrt(weight/N)
			err = tf.reshape(err, [tf.reduce_prod(tf.shape(err)),1])
		weights_jacobians = tape.jacobian(err, self.weights)
		biases_jacobians = tape.jacobian(err, self.biases)
		if biases_jacobians[-1] is None:
			biases_jacobians[-1] = tf.zeros([tf.shape(biases_jacobians[0])[0], self.output_size, 1, self.output_size])
		jacobs_tol_W = tf.squeeze(tf.concat(weights_jacobians, axis = -1))
		jacobs_tol_b = tf.squeeze(tf.concat(biases_jacobians, axis = -1))
		del tape
		return jacobs_tol_W, jacobs_tol_b, err

	# @tf.function
	def construct_Gradient(self, samples_list):
		with tf.GradientTape(persistent = True) as tape:
			loss_val = self.loss(samples_list)
		weights_grads = tape.gradient(loss_val, self.weights)
		biases_grads = tape.gradient(loss_val, self.biases)
		if biases_grads[-1] is None:
			biases_grads[-1] = tf.zeros([1, self.output_size])
		grads_tol_W = tf.transpose(tf.concat(weights_grads, axis = -1))
		grads_tol_b = tf.transpose(tf.concat(biases_grads, axis = -1))
		del tape
		return loss_val, grads_tol_W, grads_tol_b

	def construct_Hessian(self, loss_val, tape_loss):		
		num_layers = len(self.biases) 

		with tape_loss:
			loss_grad_W, loss_grad_b = self.construct_Gradient(loss_val, tape_loss)
		hessian_W = tape_loss.jacobian(loss_grad_W, self.weights)
		hessian_b = tape_loss.jacobian(loss_grad_b, self.biases)
		hessian_W = tf.squeeze(tf.concat(hessian_W, axis = 3))
		if hessian_b[-1] is None:
			hessian_b[-1] = tf.zeros([self.biases_len, self.output_size, 1, self.output_size])
		hessian_b = tf.squeeze(tf.concat(hessian_b, axis = 3))
		return hessian_W, hessian_b
		
	def update_weights_biases(self, weights_update, biases_update):
		num_layers = len(self.biases)
		W_ind_count = 0
		b_ind_count = 0 
		for l in range(num_layers):
			W_len = np.prod(self.weights_dims[l])
			b_len = np.prod(self.biases_dims[l])
			W_update = weights_update[W_ind_count:W_ind_count+W_len]
			b_update = biases_update[b_ind_count:b_ind_count+b_len]
			self.weights[l].assign_add(tf.reshape(W_update, [1, W_len]))
			self.biases[l].assign_add(tf.reshape(b_update, [1, b_len]))

			W_ind_count += W_len
			b_ind_count += b_len

	def set_weights_biases(self, new_weights, new_biases, lin = False):
		num_layers = len(self.biases) 
		if not lin:
			for l in range(num_layers):
				self.weights[l].assign(new_weights[l])
				self.biases[l].assign(new_biases[l])
		if lin:
			W_ind_count = 0
			b_ind_count = 0 
			weights = []
			biases = []
			for l in range(num_layers):
				W_len = np.prod(self.weights_dims[l])
				b_len = np.prod(self.biases_dims[l])
				W_new = new_weights[W_ind_count:W_ind_count+W_len]
				b_new = new_biases[b_ind_count:b_ind_count+b_len]
				W_new = tf.reshape(W_new, [1, W_len])
				b_new = tf.reshape(b_new, [1, b_len])
				self.weights[l].assign(W_new)
				self.biases[l].assign(b_new)
				W_ind_count += W_len
				b_ind_count += b_len
		# return self.weights, self.biases

	def get_weights_biases(self):
		weights = []
		biases = []
		num_layers = len(self.biases) 
		for l in range(num_layers):
			W = self.weights[l]
			b = self.biases[l]
			W_len = np.prod(self.weights_dims[l])
			b_len = np.prod(self.biases_dims[l])
			W_copy = tf.identity(W)
			b_copy = tf.identity(b)
			weights.append(W_copy)
			biases.append(b_copy)
			weights_lin = tf.concat((weights_lin,tf.reshape(W_copy, [1, W_len])),axis = 1) if l != 0 else tf.reshape(W_copy, [1, W_len])
			biases_lin = tf.concat((biases_lin,tf.reshape(b_copy, [1, b_len])),axis = 1) if l != 0 else tf.reshape(b_copy, [1, b_len])
		return weights, biases, weights_lin, biases_lin


	def set_weights_loss(self, samples_list, new_weights, new_biases):
		# new_weights = tf.constant(new_weights,dtype = tf.float32)
		# new_biases = tf.constant(new_biases,dtype = tf.float32)
		self.set_weights_biases(tf.squeeze(new_weights),tf.squeeze(new_biases),lin=True)
		loss_val_tf = self.loss(samples_list)
		return loss_val_tf.numpy()

	def line_search(self, samples_list, x, y, delta_x, delta_y ,tau, alpha = 0.01):
		incre_x = tau*delta_x
		incre_y = tau*delta_y
		x_new = x + incre_x
		y_new = y + incre_y
		
		self.set_weights_biases((x),(y),lin=True)
		lossval = self.loss(samples_list)
		rhs = lossval - alpha*np.sum([np.sum(incre_x*delta_x),np.sum(incre_y*delta_y)])
		self.set_weights_biases((x_new),(y_new),lin=True)
		newlossval = self.loss(samples_list)
		while newlossval>rhs:
			tau = 0.5*tau
			incre_x = tau*delta_x
			incre_y = tau*delta_y
			x_new = x + incre_x
			y_new = y + incre_y
			# self.set_weights_biases((x),(y),lin=True)
			# lossval = self.loss(samples_list)
			rhs = lossval - alpha*np.sum([np.sum(incre_x*delta_x),np.sum(incre_y*delta_y)])
			self.set_weights_biases((x_new),(y_new),lin=True)
			newlossval = self.loss(samples_list)
		return tau, tf.reshape(x_new,[1,len(x_new)]), tf.reshape(y_new,[1,len(y_new)])

	def select_batch(self, samples_list, batch_size = 64): 
		sample_batch_list = []
		for i in range(len(samples_list)):
			sample_i = samples_list[i]
			x_i = sample_i["x_tf"]
			y_i = sample_i["y_tf"]
			t_i = sample_i["t_tf"]
			xi_i = sample_i["xi_tf"]
			w1_i = sample_i["w1_tf"]
			w2_i = sample_i["w2_tf"]
			t1_i = sample_i["t1_tf"]
			t2_i = sample_i["t2_tf"]
			output_i = sample_i["target"]
			N = sample_i["N"]
			weight = sample_i["weight"]
			type_str = sample_i["type"]

			if N>batch_size:
				N_sel = np.random.choice(tf.cast(N, tf.int32),batch_size)
				x_batch = tf.constant(tf.gather(x_i,N_sel), name = "batch_x")
				y_batch = tf.constant(tf.gather(y_i,N_sel), name = "batch_y")
				t_batch = tf.constant(tf.gather(t_i,N_sel), name = "batch_t")
				xi_batch = tf.constant(tf.gather(xi_i,N_sel), name = "batch_xi")

				N_sel = N_sel.reshape([batch_size,1])
				target = tf.constant(tf.gather_nd(output_i,N_sel), name = "batch_output")
				w1_batch = tf.constant(tf.gather_nd(w1_i,N_sel), name = "batch_w1")
				w2_batch = tf.constant(tf.gather_nd(w2_i,N_sel), name = "batch_w2")
				t1_batch = tf.constant(tf.gather_nd(t1_i,N_sel), name = "batch_t1")
				t2_batch = tf.constant(tf.gather_nd(t2_i,N_sel), name = "batch_t2")

				N = tf.constant(batch_size, dtype = tf.float32)
				sample_dict = {'x_tf':x_batch, 'y_tf':y_batch, 't_tf':t_batch, 'xi_tf':xi_batch, 'w1_tf':w1_batch, 'w2_tf':w2_batch, 't1_tf':t1_batch, 't2_tf':t2_batch, 'target':target, 'N':N, 'weight':weight, 'type':type_str}
				sample_batch_list.append(sample_dict)

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

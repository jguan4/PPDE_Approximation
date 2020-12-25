import numpy as np
import time
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.random.set_seed(5)

def train_lm(net, samples_list, max_iter, tol, mu, beta, path_weight = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)

	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	f = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5:# and loss_diff != 0:
		tau = 1
		start_time = time.time()
		temp_loss, gradient, mu = update_weights(net,samples_list,mu,beta,batch_lim)

		iteration_time = time.time()-start_time

		loss_diff = loss_val-temp_loss
		loss_val = temp_loss

		if epoch%10 ==0:
			net.save_weights_biases(path_weight)
		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu is ",mu,", tau is ",tau,", time: ",iteration_time,"\n")
		if mu>1e4:
			mu = 1e-2
	net.save_weights_biases(path_weight)	

def update_weights(net, samples_list ,mu, beta, batch_lim):
	tau = 1
	m = 1
	old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases()

	_ = net.loss(samples_list)
	loss_val_tf, grads_W_list, grads_b_list = net.construct_Gradient(samples_list)
	loss_val = loss_val_tf.numpy()
	gradient = tf.norm(grads_W_list)+tf.norm(grads_b_list)
	gradient = gradient.numpy()

	JJ_W_list, JJ_b_list, Je_W,Je_b = construct_tol_Jacobian(net, samples_list, batch_lim)
	W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, mu)

	net.update_weights_biases(tf.squeeze(-tau*W_update_tol), tf.squeeze(-tau*b_update_tol))
	temp_loss_tf = net.loss(samples_list)
	temp_loss = temp_loss_tf.numpy()

	loss_diff = loss_val - temp_loss

	if loss_diff>=0:
		loss_val = temp_loss
		loss_diff1 = loss_diff
		best_loss = temp_loss
		new_weights, new_biases, new_weights_lin, new_biases_lin = net.get_weights_biases()
		best_weights = new_weights
		best_biases = new_biases
		temp_mu = mu
		while loss_diff1>0:
			net.set_weights_biases(old_weights, old_biases)
			temp_mu = np.max([1e-10,temp_mu/beta])
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, temp_mu)

			net.update_weights_biases(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = loss_val - temp_loss1
			if temp_loss1<best_loss:
				best_weights, best_biases, _, _ = net.get_weights_biases()
				best_loss = temp_loss1
		net.set_weights_biases(best_weights, best_biases)
		loss_diff = loss_val - best_loss
		loss_val = best_loss
		mu = temp_mu

	else:
		while loss_diff<0 and mu<1e16:# and m<=10:
			mu = np.min([1e16,mu*beta])
			net.set_weights_biases(old_weights, old_biases)
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, mu)
			net.update_weights_biases(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff = loss_val - temp_loss1
			m+=1

		loss_diff1 = loss_diff
		best_loss = temp_loss1
		new_weights, new_biases, new_weights_lin, new_biases_lin = net.get_weights_biases()
		best_weights = new_weights
		best_biases = new_biases
		temp_mu = mu

		while loss_diff1>=0 and temp_mu<1e5:
			net.set_weights_biases(old_weights, old_biases)
			temp_mu = np.min([1e16,temp_mu*beta])
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, temp_mu)

			net.update_weights_biases(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = best_loss - temp_loss1
			if temp_loss1<best_loss:
				best_weights, best_biases, _, _ = net.get_weights_biases()
				best_loss = temp_loss1
		net.set_weights_biases(best_weights, best_biases)
		loss_diff = loss_val - best_loss
		loss_val = best_loss
		mu = temp_mu
	return loss_val, gradient, mu

def compute_W_b_update(net, JJ_W, JJ_b, grads_W_list, grads_b_list, mu):
	grads_W_tol = grads_W_list 
	grads_b_tol = grads_b_list 

	JJ_W = JJ_W+mu*tf.eye(net.weights_len)
	JJ_b = JJ_b+mu*tf.eye(net.biases_len)
	W_update_tol = tf.linalg.solve(JJ_W, grads_W_tol)
	b_update_tol = tf.linalg.solve(JJ_b, grads_b_tol)

	return W_update_tol, b_update_tol

def compute_matrix_cond(A):
	s = tf.linalg.svd(A, full_matrices=False, compute_uv=False)
	cond = s[0]/s[-1]
	return cond

def construct_tol_Jacobian(net, samples_list, batch_lim):

	JJ_W = tf.zeros([net.weights_len, net.weights_len])
	JJ_b = tf.zeros([net.biases_len, net.biases_len])

	Je_W = tf.zeros([net.weights_len, 1])
	Je_b = tf.zeros([net.biases_len, 1])

	for i in range(len(samples_list)):
		sample_i = samples_list[i]

		Ntr_name_i = sample_i["type"]
		N_int = tf.dtypes.cast(sample_i["N"], tf.int32)
		if N_int<batch_lim:
			x_i = sample_i["x_tf"]
			y_i = sample_i["y_tf"]
			t_i = sample_i["t_tf"]
			xi_i = sample_i["xi_tf"]
			output_i = sample_i["target"]
			N = sample_i["N"]
			weight = sample_i["weight"]
	
			if Ntr_name_i == "Res":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_residual(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "B_N":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_neumann(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Init":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_solution(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "B_D":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_dirichlet(x_i, y_i, t_i, xi_i, output_i, N, weight)

			JJ_W1, JJ_b1, Je_W1, Je_b1 = compute_JJ(jacobs_tol_W, jacobs_tol_b, err_batch)
			JJ_W = JJ_W + JJ_W1
			JJ_b = JJ_b + JJ_b1
			Je_W = Je_W + Je_W1
			Je_b = Je_b + Je_b1
			
		else:
			x_i = sample_i["x_tf"]
			y_i = sample_i["y_tf"]
			t_i = sample_i["t_tf"]
			xi_i = sample_i["xi_tf"]
			output_i = sample_i["target"]
			N = sample_i["N"]
			weight = sample_i["weight"]

			for j in range(tf.dtypes.cast(N_int/batch_lim, tf.int32)):
				start_ind = j*batch_lim
				end_ind = tf.dtypes.cast(tf.math.minimum(N_int, (j+1)*batch_lim), tf.int32)
				target = output_i[start_ind:end_ind, 0:1]
				x_batch = x_i[start_ind:end_ind, 0:1]
				y_batch = y_i[start_ind:end_ind, 0:1]
				t_batch = t_i[start_ind:end_ind, 0:1]
				xi_batch = xi_i[start_ind:end_ind, 0::]

				if Ntr_name_i == "Res":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_residual(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "B_N":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_neumann(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Init":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_solution(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "B_D":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_dirichlet(x_i, y_i, t_i, xi_i, output_i, N, weight)

				JJ_W1, JJ_b1, Je_W1, Je_b1 = compute_JJ(jacobs_tol_W, jacobs_tol_b, err_batch)
				JJ_W = JJ_W + JJ_W1
				JJ_b = JJ_b + JJ_b1
				Je_W = Je_W + Je_W1
				Je_b = Je_b + Je_b1

	return JJ_W, JJ_b, Je_W, Je_b

@tf.function
def compute_JJ(jacobs_tol_W, jacobs_tol_b, err_batch):
	JJ_W = tf.matmul(jacobs_tol_W, jacobs_tol_W, transpose_a = True)
	JJ_b = tf.matmul(jacobs_tol_b, jacobs_tol_b, transpose_a = True)
	Je_W = tf.matmul(jacobs_tol_W, err_batch, transpose_a = True)
	Je_b = tf.matmul(jacobs_tol_b, err_batch, transpose_a = True)
	return JJ_W, JJ_b, Je_W, Je_b
import numpy as np
import time
import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.random.set_seed(5)

from .utils import *


def compute_W_b_update(net, JJ_W, JJ_b, grads_W_list, grads_b_list, mu):
	grads_W_tol = grads_W_list 
	grads_b_tol = grads_b_list 

	JJ_W = JJ_W+(mu+net.regular_alphas)*tf.eye(net.weights1_len)#W_reg_matrix
	JJ_b = JJ_b+(mu+net.regular_alphas)*tf.eye(net.biases1_len)
	W_update_tol = tf.linalg.solve(JJ_W, grads_W_tol)#/(np.sum(num_batches)*net.weights_len))
	b_update_tol = tf.linalg.solve(JJ_b, grads_b_tol)#/(np.sum(num_batches)*net.weights_len))

	return W_update_tol, b_update_tol

def compute_matrix_cond(A):
	s = tf.linalg.svd(A, full_matrices=False, compute_uv=False)
	cond = s[0]/s[-1]
	return cond

def construct_tol_Jacobian_uhat(net, samples_list, batch_lim):

	JJ_W = tf.zeros([net.weights1_len, net.weights1_len])
	JJ_b = tf.zeros([net.biases1_len, net.biases1_len])

	Je_W = tf.zeros([net.weights1_len, 1])
	Je_b = tf.zeros([net.biases1_len, 1])

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
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_residual_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "B_N":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_neumann_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_solution_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Reduced":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_reduced_residual(x_i, y_i, t_i, xi_i, output_i, N, weight)

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
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_residual_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "B_N":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_neumann_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_solution_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Reduced":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_reduced_residual(x_batch, y_batch, t_batch, xi_batch, target, N, weight)

				JJ_W1, JJ_b1, Je_W1, Je_b1 = compute_JJ(jacobs_tol_W, jacobs_tol_b, err_batch)
				JJ_W = JJ_W + JJ_W1
				JJ_b = JJ_b + JJ_b1
				Je_W = Je_W + Je_W1
				Je_b = Je_b + Je_b1

	return JJ_W, JJ_b, Je_W, Je_b

def construct_tol_Jacobian_b(net, samples_list, batch_lim):

	JJ_W = tf.zeros([net.weights2_len, net.weights2_len])
	JJ_b = tf.zeros([net.biases2_len, net.biases2_len])

	Je_W = tf.zeros([net.weights2_len, 1])
	Je_b = tf.zeros([net.biases2_len, 1])

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
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_residual_b(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "B_N":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_neumann_b(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_solution_b(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Reduced":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_reduced_residual(x_i, y_i, t_i, xi_i, output_i, N, weight)

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
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_residual_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "B_N":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_neumann_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_solution_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Reduced":
					jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_reduced_residual(x_batch, y_batch, t_batch, xi_batch, target, N, weight)

				JJ_W1, JJ_b1, Je_W1, Je_b1 = compute_JJ(jacobs_tol_W, jacobs_tol_b, err_batch)
				JJ_W = JJ_W + JJ_W1
				JJ_b = JJ_b + JJ_b1
				Je_W = Je_W + Je_W1
				Je_b = Je_b + Je_b1

	return JJ_W, JJ_b, Je_W, Je_b

def construct_tol_Jacobian_tol(net, samples_list, batch_lim):
	JJ_W1 = tf.zeros([net.weights1_len, net.weights1_len])
	JJ_b1 = tf.zeros([net.biases1_len, net.biases1_len])

	Je_W1 = tf.zeros([net.weights1_len, 1])
	Je_b1 = tf.zeros([net.biases1_len, 1])

	JJ_W2 = tf.zeros([net.weights2_len, net.weights2_len])
	JJ_b2 = tf.zeros([net.biases2_len, net.biases2_len])

	Je_W2 = tf.zeros([net.weights2_len, 1])
	Je_b2 = tf.zeros([net.biases2_len, 1])

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
				jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_residual_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
				jacobs_tol_W1b, jacobs_tol_b1b, jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_residual_b_tol(x_i, y_i, t_i, xi_i, output_i, N, weight)
				JJ_Wb1, JJ_bb1, Je_Wb1, Je_bb1 = compute_JJ(jacobs_tol_W1b, jacobs_tol_b1b, err_batchb)
				JJ_W1 = JJ_W1 + JJ_Wb1
				JJ_b1 = JJ_b1 + JJ_bb1
				Je_W1 = Je_W1 + Je_Wb1
				Je_b1 = Je_b1 + Je_bb1
			elif Ntr_name_i == "B_N":
				jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_neumann_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
				jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_neumann_b(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
				jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_solution_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
				jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_solution_b(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Reduced":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_reduced_residual(x_i, y_i, t_i, xi_i, output_i, N, weight)

			JJ_Wu1, JJ_bu1, Je_Wu1, Je_bu1 = compute_JJ(jacobs_tol_W1, jacobs_tol_b1, err_batchu)
			JJ_W1 = JJ_W1 + JJ_Wu1
			JJ_b1 = JJ_b1 + JJ_bu1
			Je_W1 = Je_W1 + Je_Wu1
			Je_b1 = Je_b1 + Je_bu1

			JJ_Wb2, JJ_bb2, Je_Wb2, Je_bb2 = compute_JJ(jacobs_tol_W2, jacobs_tol_b2, err_batchb)
			JJ_W2 = JJ_W2 + JJ_Wb1
			JJ_b2 = JJ_b2 + JJ_bb1
			Je_W2 = Je_W2 + Je_Wb1
			Je_b2 = Je_b2 + Je_bb1
			
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
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_residual_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					jacobs_tol_W1b, jacobs_tol_b1b, jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_residual_b_tol(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					JJ_Wb1, JJ_bb1, Je_Wb1, Je_bb1 = compute_JJ(jacobs_tol_W1b, jacobs_tol_b1b, err_batchb)
					JJ_W1 = JJ_W1 + JJ_Wb1
					JJ_b1 = JJ_b1 + JJ_bb1
					Je_W1 = Je_W1 + Je_Wb1
					Je_b1 = Je_b1 + Je_bb1
				elif Ntr_name_i == "B_N":
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_neumann_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_neumann_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_solution_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_solution_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Reduced":
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_reduced_residual(x_batch, y_batch, t_batch, xi_batch, target, N, weight)

				JJ_Wu1, JJ_bu1, Je_Wu1, Je_bu1 = compute_JJ(jacobs_tol_W1, jacobs_tol_b1, err_batchu)
				JJ_W1 = JJ_W1 + JJ_Wu1
				JJ_b1 = JJ_b1 + JJ_bu1
				Je_W1 = Je_W1 + Je_Wu1
				Je_b1 = Je_b1 + Je_bu1

				JJ_Wb2, JJ_bb2, Je_Wb2, Je_bb2 = compute_JJ(jacobs_tol_W2, jacobs_tol_b2, err_batchb)
				JJ_W2 = JJ_W2 + JJ_Wb1
				JJ_b2 = JJ_b2 + JJ_bb1
				Je_W2 = Je_W2 + Je_Wb1
				Je_b2 = Je_b2 + Je_bb1

	return JJ_W1, JJ_b1, Je_W1, Je_b1, JJ_W2, JJ_b2, Je_W2, Je_b2

def construct_tol_Jacobian_BD(net, samples_list, batch_lim):
	JJ_W1 = tf.zeros([net.weights1_len, net.weights1_len])
	JJ_b1 = tf.zeros([net.biases1_len, net.biases1_len])

	Je_W1 = tf.zeros([net.weights1_len, 1])
	Je_b1 = tf.zeros([net.biases1_len, 1])

	JJ_W2 = tf.zeros([net.weights2_len, net.weights2_len])
	JJ_b2 = tf.zeros([net.biases2_len, net.biases2_len])

	Je_W2 = tf.zeros([net.weights2_len, 1])
	Je_b2 = tf.zeros([net.biases2_len, 1])

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
	
			if Ntr_name_i == "Init" or Ntr_name_i == "B_D":
				jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_solution_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
				jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_solution_b(x_i, y_i, t_i, xi_i, output_i, N, weight)

				JJ_Wu1, JJ_bu1, Je_Wu1, Je_bu1 = compute_JJ(jacobs_tol_W1, jacobs_tol_b1, err_batchu)
				JJ_W1 = JJ_W1 + JJ_Wu1
				JJ_b1 = JJ_b1 + JJ_bu1
				Je_W1 = Je_W1 + Je_Wu1
				Je_b1 = Je_b1 + Je_bu1

				JJ_Wb2, JJ_bb2, Je_Wb2, Je_bb2 = compute_JJ(jacobs_tol_W2, jacobs_tol_b2, err_batchb)
				JJ_W2 = JJ_W2 + JJ_Wb2
				JJ_b2 = JJ_b2 + JJ_bb2
				Je_W2 = Je_W2 + Je_Wb2
				Je_b2 = Je_b2 + Je_bb2
			
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
				
				if Ntr_name_i == "Init" or Ntr_name_i == "B_D":
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_solution_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_solution_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)

					JJ_Wu1, JJ_bu1, Je_Wu1, Je_bu1 = compute_JJ(jacobs_tol_W1, jacobs_tol_b1, err_batchu)
					JJ_W1 = JJ_W1 + JJ_Wu1
					JJ_b1 = JJ_b1 + JJ_bu1
					Je_W1 = Je_W1 + Je_Wu1
					Je_b1 = Je_b1 + Je_bu1

					JJ_Wb2, JJ_bb2, Je_Wb2, Je_bb2 = compute_JJ(jacobs_tol_W2, jacobs_tol_b2, err_batchb)
					JJ_W2 = JJ_W2 + JJ_Wb2
					JJ_b2 = JJ_b2 + JJ_bb2
					Je_W2 = Je_W2 + Je_Wb2
					Je_b2 = Je_b2 + Je_bb2

	return JJ_W1, JJ_b1, Je_W1, Je_b1, JJ_W2, JJ_b2, Je_W2, Je_b2

def construct_tol_Jacobian_tol1(net, samples_list, batch_lim):
	JJ_W1 = tf.zeros([net.weights1_len, net.weights1_len])
	JJ_b1 = tf.zeros([net.biases1_len, net.biases1_len])

	Je_W1 = tf.zeros([net.weights1_len, 1])
	Je_b1 = tf.zeros([net.biases1_len, 1])

	JJ_W2 = tf.zeros([net.weights2_len, net.weights2_len])
	JJ_b2 = tf.zeros([net.biases2_len, net.biases2_len])

	Je_W2 = tf.zeros([net.weights2_len, 1])
	Je_b2 = tf.zeros([net.biases2_len, 1])

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
				jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_residual_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
				jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_residual_b(x_i, y_i, t_i, xi_i, output_i, N, weight)

			elif Ntr_name_i == "B_N":
				jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_neumann_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
				jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_neumann_b(x_i, y_i, t_i, xi_i, output_i, N, weight)
			elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
				jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_solution_uhat(x_i, y_i, t_i, xi_i, output_i, N, weight)
				jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_solution_b(x_i, y_i, t_i, xi_i, output_i, N, weight)
				
			elif Ntr_name_i == "Reduced":
				jacobs_tol_W, jacobs_tol_b, err_batch = net.construct_Jacobian_reduced_residual(x_i, y_i, t_i, xi_i, output_i, N, weight)

			JJ_Wu1, JJ_bu1, Je_Wu1, Je_bu1 = compute_JJ(jacobs_tol_W1, jacobs_tol_b1, err_batchu)
			JJ_W1 = JJ_W1 + JJ_Wu1
			JJ_b1 = JJ_b1 + JJ_bu1
			Je_W1 = Je_W1 + Je_Wu1
			Je_b1 = Je_b1 + Je_bu1

			JJ_Wb2, JJ_bb2, Je_Wb2, Je_bb2 = compute_JJ(jacobs_tol_W2, jacobs_tol_b2, err_batchb)
			JJ_W2 = JJ_W2 + JJ_Wb2
			JJ_b2 = JJ_b2 + JJ_bb2
			Je_W2 = Je_W2 + Je_Wb2
			Je_b2 = Je_b2 + Je_bb2
			
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
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_residual_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_residual_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)

				elif Ntr_name_i == "B_N":
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_neumann_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_neumann_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
				elif Ntr_name_i == "Init" or Ntr_name_i == "B_D":
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_solution_uhat(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					jacobs_tol_W2, jacobs_tol_b2, err_batchb = net.construct_Jacobian_solution_b(x_batch, y_batch, t_batch, xi_batch, target, N, weight)
					
				elif Ntr_name_i == "Reduced":
					jacobs_tol_W1, jacobs_tol_b1, err_batchu = net.construct_Jacobian_reduced_residual(x_batch, y_batch, t_batch, xi_batch, target, N, weight)

				JJ_Wu1, JJ_bu1, Je_Wu1, Je_bu1 = compute_JJ(jacobs_tol_W1, jacobs_tol_b1, err_batchu)
				JJ_W1 = JJ_W1 + JJ_Wu1
				JJ_b1 = JJ_b1 + JJ_bu1
				Je_W1 = Je_W1 + Je_Wu1
				Je_b1 = Je_b1 + Je_bu1

				JJ_Wb2, JJ_bb2, Je_Wb2, Je_bb2 = compute_JJ(jacobs_tol_W2, jacobs_tol_b2, err_batchb)
				JJ_W2 = JJ_W2 + JJ_Wb2
				JJ_b2 = JJ_b2 + JJ_bb2
				Je_W2 = Je_W2 + Je_Wb2
				Je_b2 = Je_b2 + Je_bb2

	return JJ_W1, JJ_b1, Je_W1, Je_b1, JJ_W2, JJ_b2, Je_W2, Je_b2

@tf.function
def compute_JJ(jacobs_tol_W, jacobs_tol_b, err_batch):
	JJ_W = tf.matmul(jacobs_tol_W, jacobs_tol_W, transpose_a = True)
	JJ_b = tf.matmul(jacobs_tol_b, jacobs_tol_b, transpose_a = True)
	Je_W = tf.matmul(jacobs_tol_W, err_batch, transpose_a = True)
	Je_b = tf.matmul(jacobs_tol_b, err_batch, transpose_a = True)
	return JJ_W, JJ_b, Je_W, Je_b

def train_lm_update(net, samples_list, max_iter, tol, mu, beta, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)

	mu1 = mu
	mu2 = mu
	fval = []
	grad_val = []
	times = []
	loss_f_list = []
	loss_b_d_list = []

	f_name = "f.csv"
	grad_name = "grad.csv"
	times_name = "times.csv"
	loss_f_name = "loss_f.csv"
	loss_b_d_name = "loss_b_d.csv"
	loss_b_n_name = "loss_b_n.csv"

	if save_toggle == 1:
		f_save_name = modify_filename(path_log,f_name)
		grad_save_name = modify_filename(path_log,grad_name)
		times_save_name = modify_filename(path_log,times_name)
		loss_f_save_name = modify_filename(path_log,loss_f_name)
		loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)
		loss_b_n_save_name = modify_filename(path_log,loss_b_n_name)

	if save_toggle == 2:
		fval = np.loadtxt(f_save_name, delimiter="\n")
		fval = fval.tolist()
		grad_val = np.loadtxt(grad_save_name, delimiter="\n")
		grad_val = grad_val.tolist()
		times = np.loadtxt(times_save_name, delimiter="\n")
		times = times.tolist()
		loss_f_list = np.loadtxt(loss_f_save_name, delimiter="\n")
		loss_f_list = loss_f_list.tolist()
		loss_b_d_save_name = np.loadtxt(loss_b_d_save_name, delimiter="\n")
		loss_b_d_save_name = loss_b_d_save_name.tolist()
		loss_b_n_save_name = np.loadtxt(loss_b_n_save_name, delimiter="\n")
		loss_b_n_save_name = loss_b_n_save_name.tolist()
	
	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)
	# f = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5 and loss_diff != 0:
		tau = 1

		# BNweight = weight_decay_linear(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "Init":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()
		temp_loss_uhat, gradient_uhat, mu1, net = update_weights_uhat(net,samples_list,mu1,batch_lim)
		temp_loss_b, gradient_b, mu2, net = update_weights_b(net,samples_list,mu2,batch_lim)
		temp_loss = temp_loss_uhat + temp_loss_b
		gradient = gradient_uhat+gradient_b

		iteration_time = time.time()-start_time
		times.append(iteration_time)

		loss_diff = loss_val-temp_loss
		loss_val = temp_loss

		fval.append(loss_val)
		grad_val.append(gradient)

		if save_for_plot:
			if epoch%1 == 0:
				plot_weight_name = path_plot+"/{0}.npz".format(epoch)
				net.save_weights_biases(plot_weight_name)

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
				np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				# loss_f_list = loss_f_list + net.loss_f_list
				# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 
		if epoch%10 ==0:
			net.save_weights_biases(path_weight)
		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu1 is ",mu1,", mu2 is ",mu2,", tau is ",tau,", time: ",iteration_time,"\n")
		if mu1>1e4:
			mu1 = 1e-2
		if mu2>1e4:
			mu2 = 1e-2

	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)
	if save_toggle:
		np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
		np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
		np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
		# loss_f_list = loss_f_list + net.loss_f_list
		# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
		np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 

def train_lm_update_BD(net, samples_list, max_iter, tol, mu, beta, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)

	mu1 = mu
	mu2 = mu
	mu3 = mu
	fval = []
	grad_val = []
	times = []
	loss_f_list = []
	loss_b_d_list = []

	f_name = "f.csv"
	grad_name = "grad.csv"
	times_name = "times.csv"
	loss_f_name = "loss_f.csv"
	loss_b_d_name = "loss_b_d.csv"
	loss_b_n_name = "loss_b_n.csv"

	if save_toggle == 1:
		f_save_name = modify_filename(path_log,f_name)
		grad_save_name = modify_filename(path_log,grad_name)
		times_save_name = modify_filename(path_log,times_name)
		loss_f_save_name = modify_filename(path_log,loss_f_name)
		loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)
		loss_b_n_save_name = modify_filename(path_log,loss_b_n_name)

	if save_toggle == 2:
		fval = np.loadtxt(f_save_name, delimiter="\n")
		fval = fval.tolist()
		grad_val = np.loadtxt(grad_save_name, delimiter="\n")
		grad_val = grad_val.tolist()
		times = np.loadtxt(times_save_name, delimiter="\n")
		times = times.tolist()
		loss_f_list = np.loadtxt(loss_f_save_name, delimiter="\n")
		loss_f_list = loss_f_list.tolist()
		loss_b_d_save_name = np.loadtxt(loss_b_d_save_name, delimiter="\n")
		loss_b_d_save_name = loss_b_d_save_name.tolist()
		loss_b_n_save_name = np.loadtxt(loss_b_n_save_name, delimiter="\n")
		loss_b_n_save_name = loss_b_n_save_name.tolist()
	
	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)
	# f = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5 and loss_diff != 0:
		tau = 1

		# BNweight = weight_decay_linear(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "Init":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()
		temp_loss_uhat, gradient_uhat, mu1, net = update_weights_uhat(net,samples_list,mu1,batch_lim)
		temp_loss_b, gradient_b, mu2, net = update_weights_b(net,samples_list,mu2,batch_lim)
		temp_loss_bd, gradient_bd, mu3, net = update_weights_tol_BD(net,samples_list,mu3,batch_lim)
		temp_loss_tf = net.loss(samples_list)
		temp_loss = temp_loss_tf.numpy()
		gradient = gradient_uhat+gradient_b

		iteration_time = time.time()-start_time
		times.append(iteration_time)

		loss_diff = loss_val-temp_loss
		loss_val = temp_loss

		fval.append(loss_val)
		grad_val.append(gradient)

		if save_for_plot:
			if epoch%1 == 0:
				plot_weight_name = path_plot+"/{0}.npz".format(epoch)
				net.save_weights_biases(plot_weight_name)

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
				np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				# loss_f_list = loss_f_list + net.loss_f_list
				# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 
		if epoch%10 ==0:
			net.save_weights_biases(path_weight)
		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu1 is ",mu1,", mu2 is ",mu2,", tau is ",tau,", time: ",iteration_time,"\n")
		if mu1>1e4:
			mu1 = 1e-2
		if mu2>1e4:
			mu2 = 1e-2
		if mu3>1e4:
			mu3 = 1e-2

	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)
	if save_toggle:
		np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
		np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
		np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
		# loss_f_list = loss_f_list + net.loss_f_list
		# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
		np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 

def train_lm_update_tol(net, samples_list, max_iter, tol, mu, beta, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)

	mu1 = mu
	mu2 = mu
	fval = []
	grad_val = []
	times = []
	loss_f_list = []
	loss_b_d_list = []

	f_name = "f.csv"
	grad_name = "grad.csv"
	times_name = "times.csv"
	loss_f_name = "loss_f.csv"
	loss_b_d_name = "loss_b_d.csv"
	loss_b_n_name = "loss_b_n.csv"

	if save_toggle == 1:
		f_save_name = modify_filename(path_log,f_name)
		grad_save_name = modify_filename(path_log,grad_name)
		times_save_name = modify_filename(path_log,times_name)
		loss_f_save_name = modify_filename(path_log,loss_f_name)
		loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)
		loss_b_n_save_name = modify_filename(path_log,loss_b_n_name)

	if save_toggle == 2:
		fval = np.loadtxt(f_save_name, delimiter="\n")
		fval = fval.tolist()
		grad_val = np.loadtxt(grad_save_name, delimiter="\n")
		grad_val = grad_val.tolist()
		times = np.loadtxt(times_save_name, delimiter="\n")
		times = times.tolist()
		loss_f_list = np.loadtxt(loss_f_save_name, delimiter="\n")
		loss_f_list = loss_f_list.tolist()
		loss_b_d_save_name = np.loadtxt(loss_b_d_save_name, delimiter="\n")
		loss_b_d_save_name = loss_b_d_save_name.tolist()
		loss_b_n_save_name = np.loadtxt(loss_b_n_save_name, delimiter="\n")
		loss_b_n_save_name = loss_b_n_save_name.tolist()
	
	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)
	# f = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5 and loss_diff != 0:
		tau = 1

		# BNweight = weight_decay_linear(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "Init":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()
		temp_loss, gradient, mu1, mu2, net = update_weights_tol(net,samples_list,mu1,mu2,batch_lim)

		iteration_time = time.time()-start_time
		times.append(iteration_time)

		loss_diff = loss_val-temp_loss
		loss_val = temp_loss

		fval.append(loss_val)
		grad_val.append(gradient)

		if save_for_plot:
			if epoch%1 == 0:
				plot_weight_name = path_plot+"/{0}.npz".format(epoch)
				net.save_weights_biases(plot_weight_name)

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
				np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				# loss_f_list = loss_f_list + net.loss_f_list
				# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 
		if epoch%10 ==0:
			net.save_weights_biases(path_weight)
		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu is ",mu," tau is ",tau,", time: ",iteration_time,"\n")
		if mu1>1e4:
			mu1 = 1e-2
		if mu2>1e4:
			mu2 = 1e-2

	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)
	if save_toggle:
		np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
		np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
		np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
		# loss_f_list = loss_f_list + net.loss_f_list
		# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
		np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s')

def train_lm_update_tol2(net, samples_list, max_iter, tol, mu, beta, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)

	mu1 = mu
	mu2 = mu
	fval = []
	grad_val = []
	times = []
	loss_f_list = []
	loss_b_d_list = []

	f_name = "f.csv"
	grad_name = "grad.csv"
	times_name = "times.csv"
	loss_f_name = "loss_f.csv"
	loss_b_d_name = "loss_b_d.csv"
	loss_b_n_name = "loss_b_n.csv"

	if save_toggle == 1:
		f_save_name = modify_filename(path_log,f_name)
		grad_save_name = modify_filename(path_log,grad_name)
		times_save_name = modify_filename(path_log,times_name)
		loss_f_save_name = modify_filename(path_log,loss_f_name)
		loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)
		loss_b_n_save_name = modify_filename(path_log,loss_b_n_name)

	if save_toggle == 2:
		fval = np.loadtxt(f_save_name, delimiter="\n")
		fval = fval.tolist()
		grad_val = np.loadtxt(grad_save_name, delimiter="\n")
		grad_val = grad_val.tolist()
		times = np.loadtxt(times_save_name, delimiter="\n")
		times = times.tolist()
		loss_f_list = np.loadtxt(loss_f_save_name, delimiter="\n")
		loss_f_list = loss_f_list.tolist()
		loss_b_d_save_name = np.loadtxt(loss_b_d_save_name, delimiter="\n")
		loss_b_d_save_name = loss_b_d_save_name.tolist()
		loss_b_n_save_name = np.loadtxt(loss_b_n_save_name, delimiter="\n")
		loss_b_n_save_name = loss_b_n_save_name.tolist()
	
	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)
	# f = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5 and loss_diff != 0:
		tau = 1

		# BNweight = weight_decay_linear(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "Init":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()
		temp_loss, gradient, mu, net = update_weights_tol2(net,samples_list,mu, batch_lim)

		iteration_time = time.time()-start_time
		times.append(iteration_time)

		loss_diff = loss_val-temp_loss
		loss_val = temp_loss

		fval.append(loss_val)
		grad_val.append(gradient)

		if save_for_plot:
			if epoch%1 == 0:
				plot_weight_name = path_plot+"/{0}.npz".format(epoch)
				net.save_weights_biases(plot_weight_name)

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
				np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				# loss_f_list = loss_f_list + net.loss_f_list
				# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 
		if epoch%10 ==0:
			net.save_weights_biases(path_weight)
		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu is ",mu," tau is ",tau,", time: ",iteration_time,"\n")
		if mu>1e4:
			mu = 1e-2

	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)
	if save_toggle:
		np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
		np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
		np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
		# loss_f_list = loss_f_list + net.loss_f_list
		# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
		np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s')

def train_lm_update_tol3(net, samples_list, max_iter, tol, mu, beta, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)

	mu1 = mu
	mu2 = mu
	fval = []
	grad_val = []
	times = []
	loss_f_list = []
	loss_b_d_list = []

	f_name = "f.csv"
	grad_name = "grad.csv"
	times_name = "times.csv"
	loss_f_name = "loss_f.csv"
	loss_b_d_name = "loss_b_d.csv"
	loss_b_n_name = "loss_b_n.csv"

	if save_toggle == 1:
		f_save_name = modify_filename(path_log,f_name)
		grad_save_name = modify_filename(path_log,grad_name)
		times_save_name = modify_filename(path_log,times_name)
		loss_f_save_name = modify_filename(path_log,loss_f_name)
		loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)
		loss_b_n_save_name = modify_filename(path_log,loss_b_n_name)

	if save_toggle == 2:
		fval = np.loadtxt(f_save_name, delimiter="\n")
		fval = fval.tolist()
		grad_val = np.loadtxt(grad_save_name, delimiter="\n")
		grad_val = grad_val.tolist()
		times = np.loadtxt(times_save_name, delimiter="\n")
		times = times.tolist()
		loss_f_list = np.loadtxt(loss_f_save_name, delimiter="\n")
		loss_f_list = loss_f_list.tolist()
		loss_b_d_save_name = np.loadtxt(loss_b_d_save_name, delimiter="\n")
		loss_b_d_save_name = loss_b_d_save_name.tolist()
		loss_b_n_save_name = np.loadtxt(loss_b_n_save_name, delimiter="\n")
		loss_b_n_save_name = loss_b_n_save_name.tolist()
	
	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)
	# f = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5 and loss_diff != 0:
		tau = 1

		# BNweight = weight_decay_linear(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "Init":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()
		temp_loss, gradient, mu1, mu2, net = update_weights_tol3(net,samples_list,mu1,mu2,batch_lim)

		iteration_time = time.time()-start_time
		times.append(iteration_time)

		loss_diff = loss_val-temp_loss
		loss_val = temp_loss

		fval.append(loss_val)
		grad_val.append(gradient)

		if save_for_plot:
			if epoch%1 == 0:
				plot_weight_name = path_plot+"/{0}.npz".format(epoch)
				net.save_weights_biases(plot_weight_name)

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
				np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				# loss_f_list = loss_f_list + net.loss_f_list
				# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 
		if epoch%10 ==0:
			net.save_weights_biases(path_weight)
		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu is ",mu," tau is ",tau,", time: ",iteration_time,"\n")
		if mu1>1e4:
			mu1 = 1e-2
		if mu2>1e4:
			mu2 = 1e-2

	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)
	if save_toggle:
		np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
		np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
		np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
		# loss_f_list = loss_f_list + net.loss_f_list
		# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
		np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s')

def train_lm_update_b(net, samples_list, max_iter, tol, mu, beta, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)

	mu1 = mu
	mu2 = mu
	fval = []
	grad_val = []
	times = []
	loss_f_list = []
	loss_b_d_list = []

	f_name = "f.csv"
	grad_name = "grad.csv"
	times_name = "times.csv"
	loss_f_name = "loss_f.csv"
	loss_b_d_name = "loss_b_d.csv"
	loss_b_n_name = "loss_b_n.csv"

	if save_toggle == 1:
		f_save_name = modify_filename(path_log,f_name)
		grad_save_name = modify_filename(path_log,grad_name)
		times_save_name = modify_filename(path_log,times_name)
		loss_f_save_name = modify_filename(path_log,loss_f_name)
		loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)
		loss_b_n_save_name = modify_filename(path_log,loss_b_n_name)

	if save_toggle == 2:
		fval = np.loadtxt(f_save_name, delimiter="\n")
		fval = fval.tolist()
		grad_val = np.loadtxt(grad_save_name, delimiter="\n")
		grad_val = grad_val.tolist()
		times = np.loadtxt(times_save_name, delimiter="\n")
		times = times.tolist()
		loss_f_list = np.loadtxt(loss_f_save_name, delimiter="\n")
		loss_f_list = loss_f_list.tolist()
		loss_b_d_save_name = np.loadtxt(loss_b_d_save_name, delimiter="\n")
		loss_b_d_save_name = loss_b_d_save_name.tolist()
		loss_b_n_save_name = np.loadtxt(loss_b_n_save_name, delimiter="\n")
		loss_b_n_save_name = loss_b_n_save_name.tolist()
	
	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)
	# f = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5 and loss_diff != 0:
		tau = 1

		# BNweight = weight_decay_linear(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "Init":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()
		temp_loss_b, gradient_b, mu2, net = update_weights_b(net,samples_list,mu2,batch_lim)
		temp_loss = temp_loss_b
		gradient = gradient_b

		iteration_time = time.time()-start_time
		times.append(iteration_time)

		loss_diff = loss_val-temp_loss
		loss_val = temp_loss

		fval.append(loss_val)
		grad_val.append(gradient)

		if save_for_plot:
			if epoch%1 == 0:
				plot_weight_name = path_plot+"/{0}.npz".format(epoch)
				net.save_weights_biases(plot_weight_name)

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
				np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				# loss_f_list = loss_f_list + net.loss_f_list
				# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 
		if epoch%10 ==0:
			net.save_weights_biases(path_weight)
		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu1 is ",mu1,", mu2 is ",mu2,", tau is ",tau,", time: ",iteration_time,"\n")
		if mu1>1e4:
			mu1 = 1e-2
		if mu2>1e4:
			mu2 = 1e-2

	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)
	if save_toggle:
		np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
		np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
		np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
		# loss_f_list = loss_f_list + net.loss_f_list
		# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
		np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
		np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s')

def update_weights_tol(net, samples_list ,mu1, mu2, batch_lim):
	tau = 1
	old_weights1, old_biases1, old_weights_lin1, old_biases_lin1, old_weights2, old_biases2, old_weights_lin2, old_biases_lin2 = net.get_weights_biases_tol()

	loss_val = net.loss(samples_list, save_toggle=False)
	# loss_val_tf, grads_W_list1, grads_b_list1, grads_W_list2, grads_b_list2 = net.construct_Gradient_tol(samples_list)
	# loss_val = loss_val_tf.numpy()
	# loss_val1_tf = net.loss_uhat(samples_list, save_toggle = False)
	# loss_val2_tf = net.loss_b(samples_list, save_toggle = False)
	loss_val1_tf, grads_W_list1, grads_b_list1 = net.construct_Gradient_uhat(samples_list)
	loss_val2_tf, grads_W_list2, grads_b_list2 = net.construct_Gradient_b(samples_list)

	loss_val1 = loss_val1_tf.numpy()
	loss_val2 = loss_val2_tf.numpy()

	gradient = tf.norm(grads_W_list1)+tf.norm(grads_b_list1)+tf.norm(grads_W_list2)+tf.norm(grads_b_list2)
	gradient = gradient.numpy()

	# JJ_W_list1, JJ_b_list1, Je_W1, Je_b1, JJ_W_list2, JJ_b_list2, Je_W2, Je_b2  = construct_tol_Jacobian_tol(net, samples_list, batch_lim)
	JJ_W_list1, JJ_b_list1, Je_W1,Je_b1 = construct_tol_Jacobian_uhat(net, samples_list, batch_lim)
	JJ_W_list2, JJ_b_list2, Je_W2,Je_b2 = construct_tol_Jacobian_b(net, samples_list, batch_lim)

	loss_val1, mu1, net = test_mu_uhat(net, samples_list, old_weights1, old_biases1, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, loss_val1, mu1, tau)
	loss_val2, mu2, net = test_mu_b(net, samples_list, old_weights2, old_biases2, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, loss_val2, mu2, tau)
	loss_val = loss_val1+loss_val2
	return loss_val, gradient, mu1, mu2, net

def update_weights_tol_BD(net, samples_list ,mu3, batch_lim):
	tau = 1
	old_weights1, old_biases1, old_weights_lin1, old_biases_lin1, old_weights2, old_biases2, old_weights_lin2, old_biases_lin2 = net.get_weights_biases_tol()

	loss_val = net.loss_BD(samples_list, save_toggle=False)
	loss_val_tf, grads_W_list1, grads_b_list1, grads_W_list2, grads_b_list2 = net.construct_Gradient_BD(samples_list)

	gradient = tf.norm(grads_W_list1)+tf.norm(grads_b_list1)+tf.norm(grads_W_list2)+tf.norm(grads_b_list2)
	gradient = gradient.numpy()

	JJ_W_list1, JJ_b_list1, Je_W1, Je_b1, JJ_W_list2, JJ_b_list2, Je_W2, Je_b2  = construct_tol_Jacobian_BD(net, samples_list, batch_lim)

	loss_val, mu3, net = test_mu_BD(net, samples_list, old_weights1, old_biases1, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, old_weights2, old_biases2, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, loss_val, mu3, tau)
	return loss_val, gradient, mu3, net

def update_weights_uhat(net, samples_list ,mu, batch_lim):
	tau = 1
	old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases_uhat()

	loss_val1 = net.loss_uhat(samples_list, save_toggle=False)
	loss_val_tf, grads_W_list, grads_b_list = net.construct_Gradient_uhat(samples_list)
	loss_val = loss_val_tf.numpy()
	gradient = tf.norm(grads_W_list)+tf.norm(grads_b_list)
	gradient = gradient.numpy()
	JJ_W_list, JJ_b_list, Je_W,Je_b = construct_tol_Jacobian_uhat(net, samples_list, batch_lim)
	loss_val, mu, net = test_mu_uhat(net, samples_list, old_weights, old_biases, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, loss_val1, mu, tau)
	return loss_val, gradient, mu, net
	
def update_weights_b(net, samples_list ,mu, batch_lim):
	tau = 1
	m = 1
	old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases_b()

	loss_val1 = net.loss_b(samples_list, save_toggle=True)
	loss_val_tf, grads_W_list, grads_b_list = net.construct_Gradient_b(samples_list)
	loss_val = loss_val_tf.numpy()
	gradient = tf.norm(grads_W_list)+tf.norm(grads_b_list)
	gradient = gradient.numpy()
	JJ_W_list, JJ_b_list, Je_W,Je_b = construct_tol_Jacobian_b(net, samples_list, batch_lim)
	loss_val, mu, net = test_mu_b(net, samples_list, old_weights, old_biases, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, loss_val1, mu, tau)
	return loss_val, gradient, mu, net

def update_weights_tol2(net, samples_list ,mu, batch_lim):
	tau = 1
	old_weights1, old_biases1, old_weights_lin1, old_biases_lin1, old_weights2, old_biases2, old_weights_lin2, old_biases_lin2 = net.get_weights_biases_tol()

	loss_val = net.loss(samples_list, save_toggle=False)
	# loss_val_tf, grads_W_list1, grads_b_list1, grads_W_list2, grads_b_list2 = net.construct_Gradient_tol(samples_list)
	# loss_val = loss_val_tf.numpy()
	# loss_val1_tf = net.loss_uhat(samples_list, save_toggle = False)
	# loss_val2_tf = net.loss_b(samples_list, save_toggle = False)
	loss_val_tf, grads_W_list1, grads_b_list1, grads_W_list2, grads_b_list2 = net.construct_Gradient_tol(samples_list)

	gradient = tf.norm(grads_W_list1)+tf.norm(grads_b_list1)+tf.norm(grads_W_list2)+tf.norm(grads_b_list2)
	gradient = gradient.numpy()

	JJ_W_list1, JJ_b_list1, Je_W1, Je_b1, JJ_W_list2, JJ_b_list2, Je_W2, Je_b2  = construct_tol_Jacobian_tol(net, samples_list, batch_lim)

	loss_val, mu, net = test_mu_tol2(net, samples_list, old_weights1, old_biases1, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, old_weights2, old_biases2, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, loss_val, mu, tau)

	return loss_val, gradient, mu, net

def update_weights_tol3(net, samples_list ,mu1, mu2, batch_lim):
	tau = 1
	old_weights1, old_biases1, old_weights_lin1, old_biases_lin1, old_weights2, old_biases2, old_weights_lin2, old_biases_lin2 = net.get_weights_biases_tol()

	loss_val = net.loss(samples_list, save_toggle=False)
	# loss_val_tf, grads_W_list1, grads_b_list1, grads_W_list2, grads_b_list2 = net.construct_Gradient_tol(samples_list)
	# loss_val = loss_val_tf.numpy()
	loss_val1_tf = net.loss_uhat(samples_list, save_toggle = False)
	loss_val2_tf = net.loss_b(samples_list, save_toggle = False)
	
	loss_val_tf, grads_W_list1, grads_b_list1, grads_W_list2, grads_b_list2 = net.construct_Gradient_tol(samples_list)

	gradient = tf.norm(grads_W_list1)+tf.norm(grads_b_list1)+tf.norm(grads_W_list2)+tf.norm(grads_b_list2)
	gradient = gradient.numpy()

	JJ_W_list1, JJ_b_list1, Je_W1, Je_b1, JJ_W_list2, JJ_b_list2, Je_W2, Je_b2  = construct_tol_Jacobian_tol(net, samples_list, batch_lim)

	loss_val1, mu1, net = test_mu_uhat(net, samples_list, old_weights1, old_biases1, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, loss_val1_tf.numpy(), mu1, tau)
	loss_val2, mu2, net = test_mu_b(net, samples_list, old_weights2, old_biases2, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, loss_val2_tf.numpy(), mu2, tau)
	loss_val = loss_val1+loss_val2

	return loss_val, gradient, mu1, mu2, net


def test_mu_tol2(net, samples_list, old_weights1, old_biases1, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, old_weights2, old_biases2, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, loss_val, mu, tau):
	m = 1
	W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, mu)
	W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, mu)

	net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
	net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
	temp_loss_tf = net.loss(samples_list)
	temp_loss = temp_loss_tf.numpy()

	loss_diff = loss_val - temp_loss

	# input()
	if loss_diff>=0:
		loss_val = temp_loss
		loss_diff1 = loss_diff
		best_loss = temp_loss
		new_weights1, new_biases1, new_weights_lin1, new_biases_lin1 = net.get_weights_biases_uhat()
		new_weights2, new_biases2, new_weights_lin2, new_biases_lin2 = net.get_weights_biases_b()
		best_weights1 = new_weights1
		best_biases1 = new_biases1
		best_weights2 = new_weights2
		best_biases2 = new_biases2
		temp_mu = mu
		while loss_diff1>0:
			net.set_weights_biases_uhat(old_weights1, old_biases1)
			net.set_weights_biases_b(old_weights2, old_biases2)
			temp_mu = np.max([1e-10,temp_mu/2])

			W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, temp_mu)
			W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, temp_mu)

			net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
			net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
			temp_loss_tf1 = net.loss(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = loss_val - temp_loss1
			if temp_loss1<best_loss:
				best_weights1, best_biases1, _, _ = net.get_weights_biases_uhat()
				best_weights2, best_biases2, _, _ = net.get_weights_biases_b()
				best_loss = temp_loss1
		net.set_weights_biases_uhat(best_weights1, best_biases1)
		net.set_weights_biases_b(best_weights2, best_biases2)
		loss_diff = loss_val - best_loss
		loss_val = best_loss
		mu = temp_mu

	else:
		
		while loss_diff<0 and mu<1e16:# and m<=10:
			mu = np.min([1e16,mu*2])
			net.set_weights_biases_uhat(old_weights1, old_biases1)
			net.set_weights_biases_b(old_weights2, old_biases2)
			W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, mu)
			W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, mu)
			net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
			net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
			temp_loss_tf1 = net.loss(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff = loss_val - temp_loss1
			m+=1

		loss_diff1 = loss_diff
		best_loss = temp_loss1
		new_weights1, new_biases1, new_weights_lin1, new_biases_lin1 = net.get_weights_biases_uhat()
		new_weights2, new_biases2, new_weights_lin2, new_biases_lin2 = net.get_weights_biases_b()
		best_weights1 = new_weights1
		best_biases1 = new_biases1
		best_weights2 = new_weights2
		best_biases2 = new_biases2
		temp_mu = mu

		while loss_diff1>=0 and temp_mu<1e5:
			temp_mu = np.min([1e16,temp_mu*2])
			net.set_weights_biases_uhat(old_weights1, old_biases1)
			net.set_weights_biases_b(old_weights2, old_biases2)
			W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, temp_mu)
			W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, temp_mu)
			net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
			net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
			temp_loss_tf1 = net.loss(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = best_loss - temp_loss1
			if temp_loss1<best_loss:
				best_weights1, best_biases1, _, _ = net.get_weights_biases_uhat()
				best_weights2, best_biases2, _, _ = net.get_weights_biases_b()
				best_loss = temp_loss1
		net.set_weights_biases_uhat(best_weights1, best_biases1)
		net.set_weights_biases_b(best_weights2, best_biases2)
		loss_current = net.loss(samples_list)
		loss_val = loss_current.numpy()
		loss_diff = loss_val - best_loss
		# loss_val = best_loss
		mu = temp_mu
	return loss_val, mu, net

def test_mu_BD(net, samples_list, old_weights1, old_biases1, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, old_weights2, old_biases2, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, loss_val, mu, tau):
	m = 1
	W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, mu)
	W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, mu)

	net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
	net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
	temp_loss_tf = net.loss_BD(samples_list)
	temp_loss = temp_loss_tf.numpy()

	loss_diff = loss_val - temp_loss

	# input()
	if loss_diff>=0:
		loss_val = temp_loss
		loss_diff1 = loss_diff
		best_loss = temp_loss
		new_weights1, new_biases1, new_weights_lin1, new_biases_lin1 = net.get_weights_biases_uhat()
		new_weights2, new_biases2, new_weights_lin2, new_biases_lin2 = net.get_weights_biases_b()
		best_weights1 = new_weights1
		best_biases1 = new_biases1
		best_weights2 = new_weights2
		best_biases2 = new_biases2
		temp_mu = mu
		while loss_diff1>0:
			net.set_weights_biases_uhat(old_weights1, old_biases1)
			net.set_weights_biases_b(old_weights2, old_biases2)
			temp_mu = np.max([1e-10,temp_mu/2])

			W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, temp_mu)
			W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, temp_mu)

			net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
			net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
			temp_loss_tf1 = net.loss_BD(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = loss_val - temp_loss1
			if temp_loss1<best_loss:
				best_weights1, best_biases1, _, _ = net.get_weights_biases_uhat()
				best_weights2, best_biases2, _, _ = net.get_weights_biases_b()
				best_loss = temp_loss1
		net.set_weights_biases_uhat(best_weights1, best_biases1)
		net.set_weights_biases_b(best_weights2, best_biases2)
		loss_diff = loss_val - best_loss
		loss_val = best_loss
		mu = temp_mu

	else:
		
		while loss_diff<0 and mu<1e16:# and m<=10:
			mu = np.min([1e16,mu*2])
			net.set_weights_biases_uhat(old_weights1, old_biases1)
			net.set_weights_biases_b(old_weights2, old_biases2)
			W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, mu)
			W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, mu)
			net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
			net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
			temp_loss_tf1 = net.loss_BD(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff = loss_val - temp_loss1
			m+=1

		loss_diff1 = loss_diff
		best_loss = temp_loss1
		new_weights1, new_biases1, new_weights_lin1, new_biases_lin1 = net.get_weights_biases_uhat()
		new_weights2, new_biases2, new_weights_lin2, new_biases_lin2 = net.get_weights_biases_b()
		best_weights1 = new_weights1
		best_biases1 = new_biases1
		best_weights2 = new_weights2
		best_biases2 = new_biases2
		temp_mu = mu

		while loss_diff1>=0 and temp_mu<1e5:
			temp_mu = np.min([1e16,temp_mu*2])
			net.set_weights_biases_uhat(old_weights1, old_biases1)
			net.set_weights_biases_b(old_weights2, old_biases2)
			W_update_tol1, b_update_tol1 = compute_W_b_update(net, JJ_W_list1, JJ_b_list1, grads_W_list1, grads_b_list1, temp_mu)
			W_update_tol2, b_update_tol2 = compute_W_b_update(net, JJ_W_list2, JJ_b_list2, grads_W_list2, grads_b_list2, temp_mu)
			net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol1), tf.squeeze(-tau*b_update_tol1))
			net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol2), tf.squeeze(-tau*b_update_tol2))
			temp_loss_tf1 = net.loss_BD(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = best_loss - temp_loss1
			if temp_loss1<best_loss:
				best_weights1, best_biases1, _, _ = net.get_weights_biases_uhat()
				best_weights2, best_biases2, _, _ = net.get_weights_biases_b()
				best_loss = temp_loss1
		net.set_weights_biases_uhat(best_weights1, best_biases1)
		net.set_weights_biases_b(best_weights2, best_biases2)
		loss_current = net.loss_BD(samples_list)
		loss_val = loss_current.numpy()
		loss_diff = loss_val - best_loss
		# loss_val = best_loss
		mu = temp_mu
	return loss_val, mu, net

def test_mu_uhat(net, samples_list, old_weights, old_biases, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, loss_val, mu, tau):
	m = 1
	W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, mu)

	net.update_weights_biases_uhat(tf.squeeze(-tau*W_update_tol), tf.squeeze(-tau*b_update_tol))
	temp_loss_tf = net.loss_uhat(samples_list)
	temp_loss = temp_loss_tf.numpy()

	loss_diff = loss_val - temp_loss

	# input()
	if loss_diff>=0:
		loss_val = temp_loss
		loss_diff1 = loss_diff
		best_loss = temp_loss
		new_weights, new_biases, new_weights_lin, new_biases_lin = net.get_weights_biases_uhat()
		best_weights = new_weights
		best_biases = new_biases
		temp_mu = mu
		while loss_diff1>0:
			net.set_weights_biases_uhat(old_weights, old_biases)
			temp_mu = np.max([1e-10,temp_mu/2])
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, temp_mu)

			net.update_weights_biases_uhat(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss_uhat(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = loss_val - temp_loss1
			if temp_loss1<best_loss:
				best_weights, best_biases, _, _ = net.get_weights_biases_uhat()
				best_loss = temp_loss1
		net.set_weights_biases_uhat(best_weights, best_biases)
		loss_diff = loss_val - best_loss
		loss_val = best_loss
		mu = temp_mu

	else:
		
		while loss_diff<0 and mu<1e16:# and m<=10:
			mu = np.min([1e16,mu*2])
			net.set_weights_biases_uhat(old_weights, old_biases)
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, mu)
			net.update_weights_biases_uhat(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss_uhat(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff = loss_val - temp_loss1
			m+=1

		loss_diff1 = loss_diff
		best_loss = temp_loss1
		new_weights, new_biases, new_weights_lin, new_biases_lin = net.get_weights_biases_uhat()
		best_weights = new_weights
		best_biases = new_biases
		temp_mu = mu

		while loss_diff1>=0 and temp_mu<1e5:
			net.set_weights_biases_uhat(old_weights, old_biases)
			temp_mu = np.min([1e16,temp_mu*2])
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, temp_mu)

			net.update_weights_biases_uhat(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss_uhat(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = best_loss - temp_loss1
			if temp_loss1<best_loss:
				best_weights, best_biases, _, _ = net.get_weights_biases_uhat()
				best_loss = temp_loss1
		net.set_weights_biases_uhat(best_weights, best_biases)
		loss_current = net.loss_uhat(samples_list)
		loss_val = loss_current.numpy()
		loss_diff = loss_val - best_loss
		# loss_val = best_loss
		mu = temp_mu
	return loss_val, mu, net

def test_mu_b(net, samples_list, old_weights, old_biases, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, loss_val, mu, tau):
	m = 1
	W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, mu)
	net.update_weights_biases_b(tf.squeeze(-tau*W_update_tol), tf.squeeze(-tau*b_update_tol))
	temp_loss_tf = net.loss_b(samples_list)
	temp_loss = temp_loss_tf.numpy()

	loss_diff = loss_val - temp_loss

	# input()
	if loss_diff>=0:
		loss_val = temp_loss
		loss_diff1 = loss_diff
		best_loss = temp_loss
		new_weights, new_biases, new_weights_lin, new_biases_lin = net.get_weights_biases_b()
		best_weights = new_weights
		best_biases = new_biases
		temp_mu = mu
		while loss_diff1>0:
			net.set_weights_biases_b(old_weights, old_biases)
			temp_mu = np.max([1e-10,temp_mu/2])
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, temp_mu)

			net.update_weights_biases_b(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss_b(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = loss_val - temp_loss1
			if temp_loss1<best_loss:
				best_weights, best_biases, _, _ = net.get_weights_biases_b()
				best_loss = temp_loss1
		net.set_weights_biases_b(best_weights, best_biases)
		loss_diff = loss_val - best_loss
		loss_val = best_loss
		mu = temp_mu

	else:
		while loss_diff<0 and mu<1e16:# and m<=10:
			mu = np.min([1e16,mu*2])
			net.set_weights_biases_b(old_weights, old_biases)
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, mu)
			net.update_weights_biases_b(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss_b(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff = loss_val - temp_loss1
			m+=1

		loss_diff1 = loss_diff
		best_loss = temp_loss1
		new_weights, new_biases, new_weights_lin, new_biases_lin = net.get_weights_biases_b()
		best_weights = new_weights
		best_biases = new_biases
		temp_mu = mu

		while loss_diff1>=0 and temp_mu<1e5:
			net.set_weights_biases_b(old_weights, old_biases)
			temp_mu = np.min([1e16,temp_mu*2])
			W_update_tol, b_update_tol = compute_W_b_update(net, JJ_W_list, JJ_b_list, grads_W_list, grads_b_list, temp_mu)

			net.update_weights_biases_b(tf.squeeze(-W_update_tol), tf.squeeze(-b_update_tol))
			temp_loss_tf1 = net.loss_b(samples_list)
			temp_loss1 = temp_loss_tf1.numpy()
			loss_diff1 = best_loss - temp_loss1
			if temp_loss1<best_loss:
				best_weights, best_biases, _, _ = net.get_weights_biases_b()
				best_loss = temp_loss1
		net.set_weights_biases_b(best_weights, best_biases)
		loss_current = net.loss_b(samples_list)
		loss_val = loss_current.numpy()
		loss_diff = loss_val - best_loss
		# loss_val = best_loss
		mu = temp_mu
	return loss_val, mu, net

def weight_increase_linear(epoch):
	epoch_lim = 500
	start = 1e-4
	end = 1
	m = (end-start)/epoch_lim
	y = epoch*m+start
	weight = np.amin([y,end])
	weight= tf.constant(weight, dtype = tf.float32)
	return weight

def weight_increase_exp(epoch):
	epoch_lim = 500
	start = 1e-4
	end = 1
	m = np.log(end/start)/500
	y = start*np.exp(epoch*m)
	weight = np.amin([y,end])
	weight= tf.constant(weight, dtype = tf.float32)
	return weight

def weight_decay_linear(epoch):
	epoch_lim = 500
	start = 1
	end = 1e-4
	m = (end-start)/epoch_lim
	y = epoch*m+start
	weight = np.amax([y,end])
	weight= tf.constant(weight, dtype = tf.float32)
	return weight




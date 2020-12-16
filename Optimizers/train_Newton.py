import numpy as np
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.random.set_seed(5)

def train_Newton(net, Xs_list, us_list, max_iter, tol, mu, beta, save_toggle, path_weight = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = 200

	Xs_cons = tf.constant([])
	us = tf.constant([])
	for i in range(len(net.Ntr_t)):
		if net.Ntr_t[i]:
			Xs_cons = tf.concat((Xs_cons,Xs_list[i]),axis = 0) if tf.shape(Xs_cons).numpy()[0] != 0 else Xs_list[i]
			us = tf.concat((us,us_list[i]),axis = 0) if tf.shape(us).numpy()[0] != 0 else us_list[i]
	Xs_tf_list = []
	for z in range(len(net.env.var_list)):
		var_s = net.env.var_list[z][0]
		var_e = net.env.var_list[z][1]
		Xs_tf_list.append(tf.Variable(Xs_cons[:,var_s:var_e], trainable = True, name = "sample_{0}".format(z)))

	# Xs_tf_list = []
	# for i in range(len(net.Ntr_t)):
	# 	if net.Ntr_t[i]:
	# 		Xs_i = Xs_list[i]
	# 		us_i = us_list[i]
	# 		Ntr_name_i = net.Ntr_name[i]
	# 		for j in range(int(np.ceil(net.Ntr[i]/batch_lim))):
	# 			seg_list = []
	# 			start = j*batch_lim
	# 			end = np.min([net.Ntr[i], (j+1)*batch_lim])
	# 			for z in range(len(net.env.var_list)):
	# 				var_s = net.env.var_list[z][0]
	# 				var_e = net.env.var_list[z][1]
	# 				seg_list.append(tf.Variable(Xs_i[start:end,var_s:var_e], trainable = True, name = "sample_{0}".format(z)))
	# 			seg_list.append(us_i[start:end, 0:1])
	# 			seg_list.append(Ntr_name_i)
	# 			Xs_tf_list.append(seg_list)

	# Xs = tf.Variable(Xs_cons[:,0:1], trainable = True, name = "sample_x")
	# ts = tf.Variable(Xs_cons[:,1:2], trainable = True, name = "sample_t")

	loss_val_tf, _, _  = net.loss(Xs_tf_list, us, net.Ntr)
	# loss_val_tf, _, _ = net.loss(Xs_tf_list, net.Ntr)
	loss_val = loss_val_tf.numpy()

	f = lambda x, y: net.set_weights_loss(Xs_tf_list, us, x, y)

	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter and gradient>1e-5:
		tau = 1
		old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases()

		# loss_val_tf, _, tape_loss = net.loss(Xs_tf_list, us, net.Ntr)
		loss_val_tf, _, tape_loss = net.loss(Xs_tf_list, us, net.Ntr)
		loss_val = loss_val_tf.numpy()

		grads_tol_W, grads_tol_b = net.construct_Gradient(loss_val_tf, tape_loss)
		gradient = tf.norm(grads_tol_W) + tf.norm(grads_tol_b)
		gradient = gradient.numpy()

		Hessian_W, Hessian_b = construct_tol_Hessian(net, Xs_list, us_list)


		# JJ_W, JJ_b = net.construct_tol_Jacobian(err_list, tape_loss)

		# batch_eps = 5

		# W_ave_update = tf.zeros([net.weights_len,1])
		# b_ave_update = tf.zeros([net.biases_len,1])
		# for i in range(batch_eps):
		# 	Xs_batch, ts_batch, us_batch, num_batches= net.select_batch(Xs_list, us_list)
		# 	loss_val_batch_tf, err_batch = net.loss(Xs_batch, ts_batch, us_batch, num_batches)
		# 	# _ = net.loss(Xs_batch, us_batch, num_batches)
		# 	grads_tol_W, grads_tol_b = net.construct_Gradient(loss_val_batch_tf)

		# 	gradient = tf.norm(grads_tol_W) + tf.norm(grads_tol_b)
		# 	gradient = gradient.numpy()

		# 	# with tf.device('/CPU:0'):
		# 	jacobs_tol_W, jacobs_tol_b = net.construct_Jacobian(err_batch)
		# 	# jacobs_tol_W = jacobs_tol_W/np.sum(num_batches)
		# 	# jacobs_tol_b = jacobs_tol_b/np.sum(num_batches)

		# 	JJ_W = tf.matmul(jacobs_tol_W, jacobs_tol_W, transpose_a = True)#/(np.sum(num_batches))
		# 	JJ_b = tf.matmul(jacobs_tol_b, jacobs_tol_b, transpose_a = True)#/(np.sum(num_batches))
		W_update = tf.linalg.solve(Hessian_W + (mu+net.regular_alphas[0])*tf.eye(net.weights_len), grads_tol_W)#/(np.sum(num_batches)*net.weights_len))
		b_update = tf.linalg.solve(Hessian_b + (mu+net.regular_alphas[0])*tf.eye(net.biases_len), grads_tol_b)#/(np.sum(num_batches)*net.biases_len))
		# 	W_ave_update = W_ave_update + W_update
		# 	b_ave_update = b_ave_update + b_update

		# new_tau, new_weights, new_biases = net.line_search(f, old_weights_lin, old_biases_lin, -W_update, -b_update ,1)
		# temp_loss = f(new_weights, new_biases)
		net.update_weights_biases(tf.squeeze(-tau*W_update), tf.squeeze(-tau*b_update))
		# net.set_weights_biases(new_weights, new_biases)
		temp_loss_tf, _, _ = net.loss(Xs_tf_list, us, net.Ntr)
		temp_loss = temp_loss_tf.numpy()

		loss_diff = np.abs(temp_loss-loss_val)
		# loss_val = temp_loss

		# if new_tau>=1:
		# 	# if loss_diff/temp_loss>1e-2:
		# 	mu = np.max([1e-10,mu/beta])
		# else:

		# 	new_tau, new_weights, new_biases = net.line_search(f, old_weights_lin, old_biases_lin, -grads_tol_W, -grads_tol_b ,1)
		# 	temp_loss = f(new_weights, new_biases)
		# 	loss_diff = np.abs(temp_loss-loss_val)
		# 	# if loss_diff/temp_loss>1e-2:
		# 	# 	loss_val = temp_loss
		# 	# 	tau = new_tau
		# 	# else:
		# 	# 	net.set_weights_biases(old_weights, old_biases)
		# 	mu = np.min([1e10,mu*(beta)])
		# 	if mu>1e3:
		# 		break
		if temp_loss<loss_val:
			loss_val = temp_loss
			counter += 1
			# if counter > 0:
			# if loss_diff/temp_loss>1e-2:
			mu = np.max([1e-10,mu/beta])
		else:
			counter = 0
			new_tau, new_weights, new_biases = net.line_search(f, old_weights_lin, old_biases_lin, -W_update, -b_update ,1)
			temp_loss = f(new_weights, new_biases)
			loss_diff = np.abs(temp_loss-loss_val)
			if loss_diff/temp_loss>1e-2:
				loss_val = temp_loss
				tau = new_tau
			else:
				net.set_weights_biases(old_weights, old_biases)
			mu = np.min([1e10,mu*(beta)])
			if mu>1e3:
				break

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)

		epoch += 1

		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", mu is ",mu,", tau is ",tau,"\n")
	net.save_weights_biases(path_weight)

def construct_tol_Hessian(net, Xs_list, us_list, batch_lim = 100):
	Hessian_W = tf.zeros([net.weights_len, net.weights_len])
	Hessian_b = tf.zeros([net.biases_len, net.biases_len])

	for i in range(len(net.Ntr_t)):
		num_batches = np.zeros([4], dtype = int)
		if net.Ntr_t[i]:

			Xs_i = Xs_list[i]
			us_i = us_list[i]
			num_batches[i] = net.Ntr[i]

			if net.Ntr[i]<batch_lim:
				us_batch = us_i
				Xs_batch_list = []
				for z in range(len(net.env.var_list)):
					var_s = net.env.var_list[z][0]
					var_e = net.env.var_list[z][1]
					Xs_batch_list.append(tf.Variable(Xs_i[:,var_s:var_e], trainable = True, name = "batch_{0}".format(z)))
				
				num_batches[i] = net.Ntr[i]
				loss_val_batch_tf, err_batch_list, tape_loss = net.loss(Xs_batch_list, us_batch, num_batches)
		
				# for z in range(len(err_batch_list)):
					# jacobs_tol_W, jacobs_tol_b = net.construct_Jacobian(err_batch_list[z], tape_loss)
				hess_tol_W, hess_tol_b = net.construct_Hessian(loss_val_batch_tf, tape_loss)

				Hessian_W = Hessian_W + hess_tol_W
				Hessian_b = Hessian_b + hess_tol_b

				print("Finishing the {0} indexed batch...".format(i))
			else:
				for j in range(int(np.ceil(net.Ntr[i]/batch_lim))):
					start_ind = j*batch_lim
					end_ind = np.min([net.Ntr[i], (j+1)*batch_lim])
					Xs_batch_list = []
					for z in range(len(net.env.var_list)):
						var_s = net.env.var_list[z][0]
						var_e = net.env.var_list[z][1]
						Xs_batch_list.append(tf.Variable(Xs_i[start_ind:end_ind,var_s:var_e], trainable = True, name = "batch_{0}".format(z)))

					us_batch = us_i[start_ind:end_ind, 0:1]
					num_batches[i] = end_ind-start_ind
					loss_val_batch_tf, err_batch_list, tape_loss = net.loss(Xs_batch_list, us_batch, num_batches)
	
					# for z in range(len(err_batch_list)):
						# jacobs_tol_W, jacobs_tol_b = net.construct_Jacobian(err_batch_list[z], tape_loss)

					hess_tol_W, hess_tol_b = net.construct_Hessian(loss_val_batch_tf, tape_loss)

					Hessian_W = Hessian_W + hess_tol_W
					Hessian_b = Hessian_b + hess_tol_b


					print("Finishing {0} out of {1} of {2} indexed batch...".format(end_ind, net.Ntr[i] ,i))
	return Hessian_W, Hessian_b

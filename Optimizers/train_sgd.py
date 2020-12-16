import numpy as np
import time
import os 
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.random.set_seed(5)

def train_sgd(net, samples_list, max_iter, tol, save_toggle, path_weight = "./temp.npz", path_log = "./Log/"):
	epoch = 0
	gradient = 1
	loss_diff = 1
	
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

	if save_toggle == 1:
		f_save_name = modify_filename(path_log,f_name)
		grad_save_name = modify_filename(path_log,grad_name)
		times_save_name = modify_filename(path_log,times_name)
		loss_f_save_name = modify_filename(path_log,loss_f_name)
		loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)

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
	
	loss_val_tf = net.loss(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)
	g = lambda x, y: net.set_weights_loss(samples_list, x, y)

	print("The beginning loss is ", loss_val)
	while loss_val>1e-9 and epoch<max_iter and gradient>1e-5:
		tau = 1
		sample_batch_list = net.select_batch(samples_list)
		f = lambda x, y: net.set_weights_loss(sample_batch_list, x, y)
		old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases()
		_ = net.loss(samples_list, save_toggle=True)
		loss_val_tf, grads_tol_W, grads_tol_b = net.construct_Gradient(samples_list)
		loss_val = loss_val_tf.numpy()

		# loss_val_tf, loss_list, _, tape_loss = net.loss(samples_list, net.Ntr, plot=True)
		# loss_val_tf, _, _, tape_loss = net.loss(sample_batch_list, batch_sizes, plot=False)

		gradient = tf.norm(grads_tol_W)+tf.norm(grads_tol_b)
		gradient = gradient.numpy()

		tau, new_weights, new_biases = net.line_search(f, old_weights_lin, old_biases_lin, -tf.squeeze(grads_tol_W), -tf.squeeze(grads_tol_b) ,tau)
		temp_loss = g(new_weights, new_biases)
		loss_diff = np.abs(temp_loss-loss_val)
		loss_val = temp_loss

		fval.append(loss_val)
		grad_val.append(gradient)

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
				np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				loss_f_list = loss_f_list + net.loss_f_list
				loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				np.savetxt(loss_f_save_name, loss_f_list, delimiter =", ", fmt ='% s') 
				np.savetxt(loss_b_d_save_name, loss_b_d_list, delimiter =", ", fmt ='% s') 

		epoch += 1
		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient, " tau is ",tau,"\n")
	net.save_weights_biases(path_weight)
	p.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
	np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
	np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
	loss_f_list = loss_f_list + net.loss_f_list
	loss_b_d_list = loss_b_d_list + net.loss_b_d_list
	np.savetxt(loss_f_save_name, loss_f_list, delimiter =", ", fmt ='% s') 
	np.savetxt(loss_b_d_save_name, loss_b_d_list, delimiter =", ", fmt ='% s') 

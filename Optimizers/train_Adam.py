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

def train_Adam(net, samples_list, max_iter, tol, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	loss_val = 1
	batch_lim = tf.constant(50, dtype = tf.int32)
	
	mw = tf.zeros([net.weights_len,1], tf.float32)
	mb = tf.zeros([net.biases_len,1], tf.float32)
	vw = tf.zeros([net.weights_len,1], tf.float32)
	vb = tf.zeros([net.biases_len,1], tf.float32)

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
	f = lambda x, y: net.set_weights_loss(samples_list, x, y)



	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))
	while loss_val>tol and epoch<max_iter:
		epoch += 1
		tau = 1

		# BNweight = weight_decay_exp(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "B_N":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()
		temp_loss, gradient, net, mw, mb, vw, vb = update_weights_adam(net,samples_list,epoch,mw,mb,vw,vb)

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

		print("At epoch ",epoch," loss is ",loss_val," loss diff is ",loss_diff," gradient is ",gradient,", tau is ",tau,", time: ",iteration_time,"\n")


	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)
	np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
	np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
	np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
	# loss_f_list = loss_f_list + net.loss_f_list
	# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
	np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
	np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
	np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 

def update_weights_adam(net,samples_list,epoch,mw,mb,vw,vb,batch_size = 64):
	beta1 = 0.9
	beta2 = 0.999
	eps = 1e-8
	stepsize = getstepsize(epoch)
	sample_batch_list = random_batch(samples_list, batch_size)

	loss_val1 = net.loss(sample_batch_list, save_toggle=True)
	loss_val_tf, grads_W_list, grads_b_list = net.construct_Gradient(sample_batch_list)
	loss_val = loss_val_tf.numpy()
	gradient = tf.norm(grads_W_list)+tf.norm(grads_b_list)
	gradient = gradient.numpy()
	mw = beta1*mw+(1-beta1)*grads_W_list
	mb = beta1*mb+(1-beta1)*grads_b_list

	vw = beta2*vw+(1-beta2)*grads_W_list**2
	vb = beta2*vb+(1-beta2)*grads_b_list**2

	mhatw = mw/(1-beta1**epoch)
	mhatb = mb/(1-beta1**epoch)

	vhatw = vw/(1-beta2**epoch)
	vhatb = vb/(1-beta2**epoch)

	W_update_tol = -stepsize*mhatw/(tf.math.sqrt(vhatw)+eps)
	b_update_tol = -stepsize*mhatb/(tf.math.sqrt(vhatb)+eps)

	net.update_weights_biases(tf.squeeze(W_update_tol), tf.squeeze(b_update_tol))
	batch_loss = net.loss(sample_batch_list)

	temp_loss = net.loss(samples_list)
	return temp_loss.numpy(), gradient, net, mw, mb, vw, vb



def random_batch(samples_list, batch_size):
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
		weight = sample_i["weight"]

		if N>batch_size:
			N_sel = np.random.choice(int(N.numpy()),batch_size)
			target = tf.gather(output_i,N_sel)
			x_batch = tf.gather(x_i,N_sel)
			y_batch = tf.gather(y_i,N_sel)
			t_batch = tf.gather(t_i,N_sel)
			xi_batch = tf.gather(xi_i,N_sel)
			N = tf.constant(batch_size, dtype = tf.float32)
			sample_dict = {'x_tf':x_batch, 'y_tf':y_batch, 't_tf':t_batch, 'xi_tf':xi_batch, 'target':target, 'N':N, 'type':type_str, 'weight':weight}
			sample_batch_list.append(sample_dict)
		else:
			sample_batch_list.append(sample_i)
	return sample_batch_list

def getstepsize(epoch):
	alpha = 0.01
	stepsize = alpha/np.sqrt(epoch)
	return stepsize



import numpy as np
import time
import os,psutil, tracemalloc

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.random.set_seed(5)
from Environments.ifiss.utils import *
from .utils import *


def train_lbfgs_2du1(net, samples_list, max_iter, tol, save_toggle, save_for_plot, path_weight = "./temp.npz", path_log = "./Log/", path_plot = "./temp.npz"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	counter = 0
	mlim = 10
	loss_val = 1
	batch_lim = tf.constant(100, dtype = tf.int32)

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

	# if save_toggle == 1:
	# 	f_save_name = modify_filename(path_log,f_name)
	# 	grad_save_name = modify_filename(path_log,grad_name)
	# 	times_save_name = modify_filename(path_log,times_name)
	# 	loss_f_save_name = modify_filename(path_log,loss_f_name)
	# 	loss_b_d_save_name = modify_filename(path_log,loss_b_d_name)
	# 	loss_b_n_save_name = modify_filename(path_log,loss_b_n_name)

	# if save_toggle == 2:
	# 	fval = np.loadtxt(f_save_name, delimiter="\n")
	# 	fval = fval.tolist()
	# 	grad_val = np.loadtxt(grad_save_name, delimiter="\n")
	# 	grad_val = grad_val.tolist()
	# 	times = np.loadtxt(times_save_name, delimiter="\n")
	# 	times = times.tolist()
	# 	loss_f_list = np.loadtxt(loss_f_save_name, delimiter="\n")
	# 	loss_f_list = loss_f_list.tolist()
	# 	loss_b_d_save_name = np.loadtxt(loss_b_d_save_name, delimiter="\n")
	# 	loss_b_d_save_name = loss_b_d_save_name.tolist()
	# 	loss_b_n_save_name = np.loadtxt(loss_b_n_save_name, delimiter="\n")
	# 	loss_b_n_save_name = loss_b_n_save_name.tolist()
	
	old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases()

	loss_val1 = net.loss(samples_list, save_toggle=True)
	loss_val_tf, grads_tol_W, grads_tol_b = construct_tol_gradient(net,samples_list,batch_lim*10)
	loss_val = loss_val_tf.numpy()
	print("Starting training...")
	print("Beginning loss is {0}.\n".format(loss_val))

	gradient = tf.norm(grads_tol_W)+tf.norm(grads_tol_b)
	gradient = gradient.numpy()

	tau, new_weights, new_biases = net.line_search(samples_list, tf.squeeze(old_weights_lin), tf.squeeze(old_biases_lin), -tf.squeeze(grads_tol_W), -tf.squeeze(grads_tol_b) ,1)
	
	# temp_loss = net.set_weights_loss(samples_list, new_weights, new_biases)
	# net.set_weights_biases(new_weights,new_biases,lin=True)

	loss_val_tf, grads_tol_W_new, grads_tol_b_new = construct_tol_gradient(net,samples_list,batch_lim*10)
	loss_val = loss_val_tf.numpy()
	# fval.append(loss_val)

	sW = tf.reshape(new_weights - old_weights_lin,[net.weights_len,1])
	sb = tf.reshape(new_biases - old_biases_lin,[net.biases_len,1])
	yW = grads_tol_W_new - grads_tol_W
	yb = grads_tol_b_new - grads_tol_b
	rhoW = 1/(np.tensordot(sW,yW))
	rhob = 1/(np.tensordot(sb,yb))

	sWs = []
	sbs = []
	yWs = []
	ybs = []
	rhoWs = []
	rhobs = []

	if tf.norm(sW)>1e-16 and tf.norm(sb)>1e-16:
		sWs.append(sW)
		sbs.append(sb)
			
		yWs.append(yW)
		ybs.append(yb)
		
		rhoWs.append(rhoW)		
		rhobs.append(rhob)		

	grads_tol_W = grads_tol_W_new
	grads_tol_b = grads_tol_b_new
	old_weights_lin = new_weights
	old_biases_lin = new_biases
	gradient = tf.norm(grads_tol_W)+tf.norm(grads_tol_b)
	gradient = gradient.numpy()
	epoch += 1

	print("First loss is {0}, gradient is {1}, tau is {2}.\n".format(loss_val, gradient, tau))

	# tracemalloc.start()
	# snapshot1 = tracemalloc.take_snapshot()

	while loss_val>tol and gradient>1e-5 and epoch<max_iter:# and loss_diff != 0  and epoch<max_iter: 

		# BNweight = weight_decay_linear(epoch)
		# for i in range(len(samples_list)):
		# 	dict_i = samples_list[i]
		# 	name_i = dict_i["type"]
		# 	if name_i == "Init":
		# 		dict_i["weight"] = BNweight

		start_time = time.time()

		# old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases()
		m = np.amin([len(sWs), mlim])
		pW = l_bfgs_direction(grads_tol_W,m,sWs,yWs,rhoWs)
		m = np.amin([len(sbs), mlim])
		pb = l_bfgs_direction(grads_tol_b,m,sbs,ybs,rhobs)
		print('After direction:')
		print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

		# sample_batch_list = net.select_batch(samples_list)
		tau, new_weights, new_biases = net.line_search(samples_list, tf.squeeze(old_weights_lin), tf.squeeze(old_biases_lin), tf.squeeze(pW), tf.squeeze(pb) ,2**4)
		if tau < 1e-9:
			tau, new_weights, new_biases = net.line_search(samples_list,tf.squeeze(old_weights_lin), tf.squeeze(old_biases_lin), -tf.squeeze(grads_tol_W), -tf.squeeze(grads_tol_b) ,2**4)
		print('After line search:')
		print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

		temp_loss, grads_tol_W_new, grads_tol_b_new = net.construct_Gradient(samples_list)
		print('After gradient:')
		print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
		# temp_loss, grads_tol_W_new, grads_tol_b_new = construct_tol_gradient(net,samples_list,batch_lim*10)
		temp_loss = temp_loss.numpy()
		loss_diff = loss_val-temp_loss
		loss_val = temp_loss
		
		sW = tf.reshape(new_weights - old_weights_lin,[net.weights_len,1])
		sb = tf.reshape(new_biases - old_biases_lin,[net.biases_len,1])
		yW = grads_tol_W_new - grads_tol_W
		yb = grads_tol_b_new - grads_tol_b
		rhoW = 1/(np.tensordot(sW,yW))
		rhob = 1/(np.tensordot(sb,yb))
		print('After computation:')
		print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

		grads_tol_W = grads_tol_W_new
		grads_tol_b = grads_tol_b_new
		gradient = tf.norm(grads_tol_W)+tf.norm(grads_tol_b)
		gradient = gradient.numpy()

		old_weights_lin = new_weights
		old_biases_lin = new_biases
		

		if (1/rhoW)>tf.norm(sW)**2*1e-2:
			sWs.append(sW)
			yWs.append(yW)
			rhoWs.append(rhoW)		
			if len(sWs) > mlim:
				sWs.pop(0)
				yWs.pop(0)
				rhoWs.pop(0)
				
		if (1/rhob)> tf.norm(sb)**2*1e-2:
			sbs.append(sb)
			ybs.append(yb)
			rhobs.append(rhob)	
			if len(sbs) > mlim:
				sbs.pop(0)
				ybs.pop(0)
				rhobs.pop(0)
		print('After list:')
		print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

		iteration_time = time.time()-start_time

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)
		print('After save weights:')
		print(psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

		epoch += 1
		print("At epoch ",epoch,", loss is ",loss_val,", loss diff is ",loss_diff,", gradient is ",gradient, ", tau is ",tau,", iteration_time is ",iteration_time,"\n")


		# fval.append(loss_val)
		# grad_val.append(gradient)

		if save_for_plot:
			if epoch%10 == 0:
				plot_weight_name = path_plot+"/{0}.npz".format(epoch)
				net.save_weights_biases(plot_weight_name)
		# snapshot2 = tracemalloc.take_snapshot()
		# top_stats = snapshot2.compare_to(snapshot1, 'lineno')
		# print("[ Top 10 differences ]")
		# for stat in top_stats[:10]:
		# 	print(stat)
		# input()
		# snapshot1 = snapshot2

		# if save_toggle:
		# 	if epoch%10 ==0:
		# 		net.save_weights_biases(path_weight)
				# np.savetxt(f_save_name, fval, delimiter =", ", fmt ='% s') 
				# np.savetxt(grad_save_name, grad_val, delimiter =", ", fmt ='% s') 
				# np.savetxt(times_save_name, times, delimiter =", ", fmt ='% s') 
				# loss_f_list = loss_f_list + net.loss_f_list
				# loss_b_d_list = loss_b_d_list + net.loss_b_d_list
				# np.savetxt(loss_f_save_name, net.loss_f_list, delimiter =", ", fmt ='% s') 
				# np.savetxt(loss_b_d_save_name, net.loss_b_d_list, delimiter =", ", fmt ='% s') 
				# np.savetxt(loss_b_n_save_name, net.loss_b_n_list, delimiter =", ", fmt ='% s') 

	net.save_weights_biases(path_weight)
	if save_for_plot:
		plot_weight_name = path_plot+"/{0}.npz".format(epoch)
		net.save_weights_biases(plot_weight_name)


def l_bfgs_direction(g, m, ss, ys, rhos):
	q = g
	alphas = np.zeros((m))
	for j in range(m):
		i = -(j+1)
		alpha = rhos[i]*np.tensordot(ss[i],q)
		q = q - alpha*ys[i]
		alphas[i] = alpha
	gamma = np.tensordot(ss[-1], ys[-1])/np.tensordot(ys[-1], ys[-1])
	z = gamma*q
	for i in range(m):
		beta = rhos[i]*np.tensordot(ys[i],z)
		z = z + ss[i]*(alphas[i] - beta)
	z = -z
	return z

def construct_tol_gradient(net, samples_list, batch_lim):
	grads_W = tf.zeros([net.weights_len,1])
	grads_b = tf.zeros([net.biases_len,1])
	loss_vals = tf.constant(0, dtype = tf.float32)
	for i in range(len(samples_list)):
		sample_i = samples_list[i]

		Ntr_name_i = sample_i["type"]
		N_int = tf.dtypes.cast(sample_i["N"], tf.int32)
		if N_int<batch_lim:

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
	
			dicti = {'x_tf':x_i, 'y_tf':y_i, 't_tf':t_i, 'xi_tf':xi_i, 'w1_tf':w1_i, 'w2_tf':w2_i, 't1_tf':t1_i, 't2_tf':t2_i, 'target':output_i, 'N':N, 'type':Ntr_name_i, 'weight':weight}
			samplelisti = [dicti]

			loss_val, grads_tol_W, grads_tol_b = net.construct_Gradient(samplelisti)

			loss_vals = loss_vals+loss_val
			grads_W = grads_W+grads_tol_W
			grads_b = grads_b+grads_tol_b

		else:
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

			looptimes = tf.dtypes.cast(np.ceil(N_int/batch_lim), tf.int32)
			# looptimesf = tf.dtypes.cast(np.ceil(N_int/batch_lim), tf.float32)

			for j in range(looptimes):
				start_ind = j*batch_lim
				end_ind = tf.dtypes.cast(tf.math.minimum(N_int, (j+1)*batch_lim), tf.int32)
				target = output_i[start_ind:end_ind, 0::]

				x_batch = x_i[start_ind:end_ind, 0:1]
				y_batch = y_i[start_ind:end_ind, 0:1]
				t_batch = t_i[start_ind:end_ind, 0:1]
				xi_batch = xi_i[start_ind:end_ind, 0::]
				w1_batch = w1_i[start_ind:end_ind, 0::]
				w2_batch = w2_i[start_ind:end_ind, 0::]
				t1_batch = t1_i[start_ind:end_ind, 0::]
				t2_batch = t2_i[start_ind:end_ind, 0::]

				# tempsize = tf.size(x_batch,out_type=tf.float32)
				# dicti = {'x_tf':x_batch, 'y_tf':y_batch, 't_tf':t_batch, 'xi_tf':xi_batch, 'target':target, 'N':tempsize, 'type':Ntr_name_i, 'weight':weight}
				dicti = {'x_tf':x_batch, 'y_tf':y_batch, 't_tf':t_batch, 'xi_tf':xi_batch, 'w1_tf':w1_batch, 'w2_tf':w2_batch, 't1_tf':t1_batch, 't2_tf':t2_batch, 'target':target, 'N':N, 'type':Ntr_name_i, 'weight':weight}
				samplelisti = [dicti]

				loss_val, grads_tol_W, grads_tol_b = net.construct_Gradient(samplelisti)

				# loss_vals = loss_vals+loss_val*tempsize
				# grads_W = grads_W+grads_tol_W*tempsize
				# grads_b = grads_b+grads_tol_b*tempsize
				loss_vals = loss_vals+loss_val
				grads_W = grads_W+grads_tol_W
				grads_b = grads_b+grads_tol_b
			# loss_vals = loss_vals/N
			# grads_W = grads_W/N
			# grads_b = grads_b/N
	return loss_vals, grads_W, grads_b
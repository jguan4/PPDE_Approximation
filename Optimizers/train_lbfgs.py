import numpy as np
import os 
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.random.set_seed(5)

def train_lbfgs(net, samples_list, max_iter, tol, save_toggle, m=5, path_weight = "./temp.npz", path_log = "./Log/"):

	epoch = 0
	gradient = 1
	loss_diff = 1
	tau = 1

	rhoWs = []
	rhobs = []
	sWs = []
	sbs = []
	yWs = []
	ybs = []

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
	

	g = lambda x, y: net.set_weights_loss(samples_list, x, y)
	old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases()

	_ = net.loss(samples_list, save_toggle=True)
	loss_val_tf, grads_tol_W, grads_tol_b = net.construct_Gradient(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)

	gradient = tf.norm(grads_tol_W)+tf.norm(grads_tol_b)
	gradient = gradient.numpy()

	tau, new_weights, new_biases = net.line_search(g, old_weights_lin, old_biases_lin, -tf.squeeze(grads_tol_W), -tf.squeeze(grads_tol_b) ,tau)
	
	temp_loss = g(new_weights, new_biases)
	_ = net.loss(samples_list, save_toggle=True)
	loss_val_tf, grads_tol_W, grads_tol_b = net.construct_Gradient(samples_list)
	loss_val = loss_val_tf.numpy()
	fval.append(loss_val)

	sW = tf.transpose(new_weights - old_weights_lin)
	sb = tf.transpose(new_biases - old_biases_lin)

	yW = grads_tol_W_new - grads_tol_W
	yb = grads_tol_b_new - grads_tol_b
	rhoW = 1/(tf.matmul(sW,yW,transpose_a = True))
	rhob = 1/(tf.matmul(sb,yb,transpose_a = True))

	if tf.norm(sW)>1e-16 and tf.norm(sb)>1e-16:
		sWs.insert(0, sW)
		sbs.insert(0, sb)
			
		yWs.insert(0, yW)
		ybs.insert(0, yb)
		
		rhoWs.insert(0,rhoW)		
		rhobs.insert(0,rhob)		

	grads_tol_W = grads_tol_W_new
	grads_tol_b = grads_tol_b_new
	gradient = tf.norm(grads_tol_W)+tf.norm(grads_tol_b)
	gradient = gradient.numpy()

	print("The beginning loss is ", loss_val)
	while loss_val>1e-9 and epoch<max_iter and gradient>1e-5:
		tau = 1
		start_time = time.time()

		old_weights, old_biases, old_weights_lin, old_biases_lin = net.get_weights_biases()
		pW = l_bfgs_direction(grads_tol_W,sWs,yWs,rhoWs)
		pb = l_bfgs_direction(grads_tol_b,sbs,ybs,rhobs)

		tau, new_weights, new_biases = net.line_search(g, old_weights_lin, old_biases_lin, tf.squeeze(pW), tf.squeeze(pb) ,1)
		if tau < 1e-9:
			tau, new_weights, new_biases = net.line_search(g, old_weights_lin, old_biases_lin, -tf.squeeze(grads_tol_W), -tf.squeeze(grads_tol_b) ,1)

		temp_loss = g(new_weights, new_biases)
		loss_diff = np.abs(temp_loss-loss_val)
		loss_val = temp_loss

		_ = net.loss(samples_list, save_toggle=True)
		loss_val_tf, grads_tol_W, grads_tol_b = net.construct_Gradient(samples_list)
		loss_val = loss_val_tf.numpy()
		
		sW_n = tf.transpose(new_weights-old_weights_lin)
		sb_n = tf.transpose(new_biases-old_biases_lin)
		
		yW_n = grads_tol_W_new - grads_tol_W
		yb_n = grads_tol_b_new - grads_tol_b

		rhoW_n = 1/(tf.matmul(yW_n, sW_n, transpose_a = True))
		rhob_n = 1/(tf.matmul(yb_n, sb_n, transpose_a = True))

		grads_tol_W = grads_tol_W_new
		grads_tol_b = grads_tol_b_new
		gradient = tf.norm(grads_tol_W)+tf.norm(grads_tol_b)
		gradient = gradient.numpy()

		if tf.norm(sW)>1e-16 and tf.norm(sb)>1e-16:
			sWs.insert(0, sW)
			sbs.insert(0, sb)
				
			yWs.insert(0, yW)
			ybs.insert(0, yb)
			
			rhoWs.insert(0,rhoW)		
			rhobs.insert(0,rhob)	
			if len(sWs) > m:
				sWs.pop()
				sbs.pop()

				yWs.pop()
				ybs.pop()

				rhoWs.pop()
				rhobs.pop()

		iteration_time = time.time()-start_time

		if save_toggle:
			if epoch%10 ==0:
				net.save_weights_biases(path_weight)

		epoch += 1
		print("At epoch ",epoch,", loss is ",loss_val,", loss diff is ",loss_diff,", gradient is ",gradient, ", tau is ",tau,", iteration_time is ",iteration_time,"\n")
	net.save_weights_biases(path_weight)

def l_bfgs_direction(g, ss, ys, rhos):
	a = []
	m = len(ss)
	for i in range(m):
		alpha = rhos[i]*tf.matmul(ss[i],g, transpose_a = True)
		g = g - alpha*ys[i]
		a.append(alpha)
	r = tf.matmul(ss[0], ys[0], transpose_a = True)/tf.matmul(ys[0], ys[0], transpose_a = True)
	g = r*g
	for i in range(m-1, -1, -1):
		beta = rhos[i]*tf.matmul(ys[i], g, transpose_a = True)
		g = g + ss[i]*(a[i] - beta)
	p = -g
	return p
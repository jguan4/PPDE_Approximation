import numpy as np
import sys
import os 
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = ''

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
tf.random.set_seed(5)

class Loss_Vis:
	def __init__(self, net, samplelist, dimension = 2):
		self.net = net
		self.samplelist = samplelist
		self.dimension = dimension
		self.N = 101
		self.maxmin = -0.2
		self.minmax = 0.2
		self.alphas = np.linspace(self.maxmin,self.minmax,self.N)
		self.betas = np.linspace(self.maxmin,self.minmax,self.N)
		self.A_equi,self.B_equi = np.meshgrid(self.alphas,self.betas)

	def visualize_space(self):
		self.generate_direction()
		self.generate_space()
		self.plot_space()

	def visualize_path(self, folder):
		self.stored_weights_path = folder
		self.generate_diff_mat()
		self.generate_pca_direction()
		self.project_weights()
		self.generate_space()
		self.plot_space()
		self.plot_trajectory()

	def visualize_path_lev(self, folder):
		self.stored_weights_path = folder
		self.generate_diff_mat()
		self.generate_lev_direction()
		self.project_weights()
		self.plot_by_section(50)
		self.generate_space()
		self.plot_space()
		self.plot_trajectory()

	def generate_diff_mat(self):
		list_of_files  = os.listdir(self.stored_weights_path)
		paths = [os.path.join(self.stored_weights_path, basename) for basename in list_of_files]
		latest_file = max(paths, key=os.path.getctime)
		last_weight_name = os.path.basename(latest_file)
		self.last_iter = int(last_weight_name.split('.')[0])
		
		npzfile = np.load(latest_file)
		final_weight = npzfile['weights_lin']
		final_biase = npzfile['biases_lin']

		weight_diff = np.zeros([np.prod(final_weight.shape),self.last_iter])
		biases_diff = np.zeros([np.prod(final_biase.shape),self.last_iter])
		for i in range(self.last_iter):
			filename = self.stored_weights_path+"/{0}.npz".format(i)
			npzfile = np.load(filename)
			weights_lin = npzfile['weights_lin']
			biases_lin = npzfile['biases_lin']
			diffw = weights_lin-final_weight
			diffb = biases_lin - final_biase
			weight_diff[:,i] = np.squeeze(diffw)
			biases_diff[:,i] = np.squeeze(diffb)

		self.diff_mat = np.concatenate((weight_diff,biases_diff), axis=0)
		self.old_weights_lin = final_weight
		self.old_biases_lin = final_biase
		
	def generate_pca_direction(self):
		u,s,vh = np.linalg.svd(self.diff_mat,full_matrices=False)
		p = u*s

		self.direction1_weight = p[0:self.net.weights_len,0]
		self.direction1_biases = p[self.net.weights_len::,0]
		self.direction2_weight = p[0:self.net.weights_len,1]
		self.direction2_biases = p[self.net.weights_len::,1]

		self.filter_normalization()

	def generate_lev_direction(self):
		m = 100
		k = 1000
		uw, sw, vh = np.linalg.svd(self.diff_mat,full_matrices=False)
		v = vh.T
		ps = np.sum(v[:,0:k]**2,axis=1,keepdims = True)/k
		inds = np.argsort(ps,axis=0)
		inds_sel = inds[0:m]
		A = self.diff_mat[:,np.squeeze(inds_sel)]
		uw, sw, vh = np.linalg.svd(A,full_matrices=False)
		p = uw*sw

		self.direction1_weight = p[0:self.net.weights_len,0]
		self.direction1_biases = p[self.net.weights_len::,0]
		self.direction2_weight = p[0:self.net.weights_len,1]
		self.direction2_biases = p[self.net.weights_len::,1]

	def project_weights(self):
		coords = np.zeros([2,self.last_iter+1])
		self.direction1 = np.concatenate((self.direction1_weight,self.direction1_biases))
		self.direction1 = np.reshape(self.direction1,[1,len(self.direction1)])
		self.direction2 = np.concatenate((self.direction2_weight,self.direction2_biases))
		self.direction2 = np.reshape(self.direction2,[1,len(self.direction2)])
		coords1 = self.direction1@self.diff_mat
		coords2 = self.direction2@self.diff_mat
		coords[:,0:self.last_iter] = np.concatenate((coords1,coords2),axis = 0)
		# coords = coords/np.tile(np.max(np.abs(coords),axis=1),(coords.shape[1],1)).T*10

		mesh_max = np.max((coords),axis=1)
		mesh_max = 1.5*mesh_max
		mesh_max = np.maximum(mesh_max,self.minmax*np.ones(mesh_max.shape))

		mesh_min = np.min((coords),axis=1)
		mesh_min = 1.5*mesh_min
		mesh_min = np.minimum(mesh_min,self.maxmin*np.ones(mesh_min.shape))

		self.coords = coords
		self.alphas = np.linspace(mesh_min[0],mesh_max[0],self.N)
		self.betas = np.linspace(mesh_min[1],mesh_max[1],self.N)
		self.A,self.B = np.meshgrid(self.alphas,self.betas)

	def generate_direction(self):
		old_weights, old_biases, self.old_weights_lin, self.old_biases_lin = self.net.get_weights_biases()
		self.old_weights_lin = self.old_weights_lin.numpy()
		self.old_biases_lin = self.old_biases_lin.numpy()
		
		self.direction1_biases = np.random.normal(size=self.old_biases_lin.shape)
		self.direction1_weight = np.random.normal(size=self.old_weights_lin.shape)
		if self.dimension == 2:
			self.direction2_biases = np.random.normal(size=self.old_biases_lin.shape)
			self.direction2_weight = np.random.normal(size=self.old_weights_lin.shape)
		self.filter_normalization()

	def filter_normalization(self):
		W_ind_count = 0
		b_ind_count = 0 
		num_layers = len(self.net.biases) 
		for l in range(num_layers):
			W_len = np.prod(self.net.weights_dims[l])
			b_len = np.prod(self.net.biases_dims[l])
			W = self.old_weights_lin[W_ind_count:W_ind_count+W_len]
			b = self.old_biases_lin[b_ind_count:b_ind_count+b_len]
			d1w = self.direction1_weight[W_ind_count:W_ind_count+W_len]
			d1b = self.direction1_biases[b_ind_count:b_ind_count+b_len]
			self.direction1_weight[W_ind_count:W_ind_count+W_len] = d1w/np.linalg.norm(d1w)*np.linalg.norm(W)
			self.direction1_biases[b_ind_count:b_ind_count+b_len] = d1b/np.linalg.norm(d1b)*np.linalg.norm(b)
			if self.dimension == 2:
				d2w = self.direction2_weight[W_ind_count:W_ind_count+W_len]
				d2b = self.direction2_biases[b_ind_count:b_ind_count+b_len]
				self.direction2_weight[W_ind_count:W_ind_count+W_len] = d2w/np.linalg.norm(d2w)*np.linalg.norm(W)
				self.direction2_biases[b_ind_count:b_ind_count+b_len] = d2b/np.linalg.norm(d2b)*np.linalg.norm(b)
			W_ind_count += W_len
			b_ind_count += b_len

	def column_select(self,A,c,k):
		uw, sw, vh = np.linalg.svd(A,full_matrices=False)
		v = vh.T
		ps = np.sum(v[:,0:k]**2,axis=1,keepdims = True)/k
		cps = np.minimum(c*ps,np.ones(ps.shape))
		probs = np.random.rand(cps.shape[0],cps.shape[1])
		inds = [i for i in range(len(cps)) if probs[i]<cps[i]]
		C = A[:,inds]

	def generate_space(self):
		if not hasattr(self,'A'):
			self.A = self.A_equi
			self.B = self.B_equi
		self.loss_mat = np.zeros(self.A.shape)
		for i in range(self.A.shape[0]):
			for j in range(self.A.shape[1]):
				alpha = self.A[i,j]
				beta = self.B[i,j]
				if self.dimension == 1:
					pass
				elif self.dimension == 2:
					new_weights_lin = self.old_weights_lin + alpha*self.direction1_weight + beta*self.direction2_weight
					new_biases_lin = self.old_biases_lin + alpha*self.direction1_biases + beta*self.direction2_biases
				loss_val = self.net.set_weights_loss(self.samplelist, new_weights_lin, new_biases_lin)
				self.loss_mat[i,j] = loss_val

	def plot_space(self):
		loss_min = np.amin(self.loss_mat)
		fig = plt.figure()
		ax = fig.gca(projection='3d')
		surf = ax.plot_surface(self.A, self.B, np.log10(self.loss_mat), cmap=cm.coolwarm,
					   linewidth=0, antialiased=False)
		ax.set_xlabel(r'$\alpha$')
		ax.set_ylabel(r'$\beta$')
		ax.set_zlabel(r'$log_{10}(L)$')
		plt.show()

	def plot_trajectory(self):
		fig = plt.figure()
		left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
		ax = fig.add_axes([left, bottom, width, height]) 

		cp = ax.contour(self.A, self.B, np.log10(self.loss_mat),cmap=cm.coolwarm)
		ax.autoscale(False) # To avoid that the scatter changes limits
		ax.scatter(self.coords[0,0],self.coords[1,0],c='b',zorder=1)
		ax.scatter(self.coords[0,1:-1],self.coords[1,1:-1],c='r',zorder=1)
		ax.scatter(self.coords[0,-1],self.coords[1,-1],c='black',zorder=1)

		# cp = ax.contour(self.A, self.B, np.log10(self.loss_mat),cmap=cm.coolwarm)
		ax.clabel(cp, inline=True, fontsize=10)
		ax.set_xlabel(r'$\alpha$')
		ax.set_ylabel(r'$\beta$')

		plt.show()

	def plot_by_section(self,k):
		fig = plt.figure()
		for i in range(0,self.last_iter,k):
			diff_i = self.diff_mat[:,i]
			diff_weight = diff_i[0:self.net.weights_len]
			diff_biase = diff_i[self.net.weights_len::]
			coords
			self.loss_mat = np.zeros(self.A_equi.shape)
			for i in range(self.A_equi.shape[0]):
				for j in range(self.A_equi.shape[1]):
					alpha = self.A_equi[i,j]
					beta = self.B_equi[i,j]
					if self.dimension == 1:
						pass
					elif self.dimension == 2:
						new_weights_lin = self.old_weights_lin + np.squeeze(diff_weight) + alpha*self.direction1_weight + beta*self.direction2_weight
						new_biases_lin = self.old_biases_lin + np.squeeze(diff_biase) + alpha*self.direction1_biases + beta*self.direction2_biases
					loss_val = self.net.set_weights_loss(self.samplelist, new_weights_lin, new_biases_lin)
					self.loss_mat[i,j] = loss_val
			# self.plot_space()
			left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
			ax = fig.add_axes([left, bottom, width, height]) 

			cp = ax.contour(self.A, self.B, np.log10(self.loss_mat),cmap=cm.coolwarm)
			ax.autoscale(False) # To avoid that the scatter changes limits
			ax.scatter(self.coords[0,0],self.coords[1,0],c='b',zorder=1)
			ax.scatter(self.coords[0,1:-1],self.coords[1,1:-1],c='r',zorder=1)
			ax.scatter(self.coords[0,-1],self.coords[1,-1],c='black',zorder=1)

			# cp = ax.contour(self.A, self.B, np.log10(self.loss_mat),cmap=cm.coolwarm)
			ax.clabel(cp, inline=True, fontsize=10)
			ax.set_xlabel(r'$\alpha$')
			ax.set_ylabel(r'$\beta$')

			plt.show()

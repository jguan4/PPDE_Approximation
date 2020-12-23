import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt 

class CD_1D:
	def __init__(self, h):
		self.h = h
		self.N = int(1/self.h)+1
		self.nx = self.N-2
		self.X = np.linspace(0,1,num=self.N)

	def u_exact(self,xi):
		u = 1-np.exp((self.X-1)/xi)
		return u

	def generate_one_sol(self, xi):
		r = self.h/(2*xi)
		A = sparse.diags([-(1+r),2,-(1-r)],[-1,0,1],\
			shape=(self.nx,self.nx), format = 'csr')
		F = np.zeros((self.nx,1))
		F[0] = (1+r)*(1-np.exp(-1/xi))
		u = spsolve(A,F)
		U = np.zeros((self.N,1))
		U[1:-1,0] = u
		U[0] = 1-np.exp(-1/xi)
		return U

if __name__ == "__main__":
	h = 1/2048
	xi = 0.1
	env = CD_1D(h)
	U = env.generate_one_sol(xi)
	U_ex = env.u_exact(xi)
	plt.plot(env.X,U,'ro')
	plt.plot(env.X,U_ex,'b')
	plt.ylabel('u')
	plt.xlabel('x')
	plt.show()
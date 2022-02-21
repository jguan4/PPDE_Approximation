import numpy as np
from scipy.sparse import csr_matrix
from .utils import *

class Grid(object):
	def __init__(self, n):
		self.n = 2**n
		self.np = self.n//2
		# self.q_in = q_in
		self.make_uniform_grid()
		self.make_Q2_elements()
		self.find_boundary_inds()
		self.q2q1grid()
		self.stokes_q2q1()

	def make_uniform_grid(self):
		npstep = 1/self.np
		self.y = np.arange(-1,1+npstep,npstep)
		self.x = self.y
		self.X, self.Y = np.meshgrid(self.x,self.y) 
		self.nvtx = (self.n+1)**2
		self.xx = self.X.reshape(-1,1)
		self.yy = self.Y.reshape(-1,1)
		self.xy = np.hstack([self.xx,self.yy])

	def make_Q2_elements(self):
		self.mv = np.array([])
		kx = 0
		ky = 0
		self.mel = 0
		for j in range(self.np):
			for i in range(self.np):
				mref = (self.n+1)*(ky)+kx
				nvv = np.zeros([1,9],dtype = np.int16)
				nvv[0,0] = mref
				nvv[0,1] = mref+2
				nvv[0,2] = mref+2*self.n+4
				nvv[0,3] = mref+2*self.n+2
				nvv[0,4] = mref+1
				nvv[0,5] = mref+self.n+3 
				nvv[0,6] = mref+2*self.n+3 
				nvv[0,7] = mref+self.n+1
				nvv[0,8] = mref+self.n+2
				self.mv = np.vstack([self.mv,nvv]) if self.mv.size else nvv
				kx += 2
				self.mel += 1
			ky += 2
			kx = 0

	def find_boundary_inds(self):
		k1 = np.argwhere(self.xy[:,1]==-1)
		e1 = [k for k in range(self.mel) if self.mv[k,4] in k1]
		e1 =  np.atleast_2d(e1).T
		ef1 = np.ones(np.size(e1))

		k2 = np.argwhere((self.xy[:,0]==1) & (self.xy[:,1]<1) & (self.xy[:,1]>-1))
		e2 = [k for k in range(self.mel) if self.mv[k,5] in k2]
		e2 =  np.atleast_2d(e2).T
		ef2 = np.ones(np.size(e2))
		
		k3 = np.argwhere(self.xy[:,1]==1)
		e3 = [k for k in range(self.mel) if self.mv[k,6] in k3]
		e3 =  np.atleast_2d(e3).T
		ef3 = np.ones(np.size(e3))
		
		k4 = np.argwhere((self.xy[:,0]==-1) & (self.xy[:,1]<1) & (self.xy[:,1]>-1))
		e4 = [k for k in range(self.mel) if self.mv[k,7] in k4]
		e4 =  np.atleast_2d(e4).T
		ef4 = np.ones(np.size(e4))

		self.bound = np.sort(np.vstack([k1,k2,k3,k4]),axis = 0)
		# self.mbound = np.vstack([np.hstack([e1,ef1]),np.hstack([e2,ef2]),np.hstack([e3,ef3]),np.hstack([e4,ef4])])

	def q2q1grid(self):
		mmv = np.reshape(self.mv[:,0:4].T, [4*self.mel])
		self.mapp = np.unique(np.sort(mmv))
		mmp = np.zeros([4*self.mel,1])
		for jj in range(4*self.mel):
			mmp[jj] = np.argwhere(self.mapp == mmv[jj])
		self.mp = np.reshape(mmp,[4,self.mel]).T
		self.xyp = self.xy[self.mapp,:]
		self.Xp = np.reshape(self.xyp[:,0], [self.np+1,self.np+1])
		self.Yp = np.reshape(self.xyp[:,1], [self.np+1,self.np+1])

	def stokes_q2q1(self):

		nngpt=9      
		s = np.zeros(nngpt)
		t = np.zeros(nngpt)
		wt = np.zeros(nngpt)

		x=self.xy[:,0]  
		y=self.xy[:,1] 
		
		xp=self.xyp[:,0]  
		yp=self.xyp[:,1] 
		
		nvtx=len(self.xx)  
		nu=2*nvtx  
		npp=len(xp)  
		nel=len(self.mv[:,0]) 

		lx=np.amax(self.xx)-np.amin(self.xx)  
		ly=np.amax(self.yy)-np.amin(self.yy) 
		hx=np.amax(np.diff(self.xx.flatten()))  
		hy=np.amax(np.diff(self.yy.flatten())) 

		self.A = csr_matrix((nu,nu)) 
		self.G = csr_matrix((nu,nu)) 
		self.Bx = csr_matrix((nvtx,nvtx)) 
		self.By = csr_matrix((nvtx,nvtx)) 
		bx = csr_matrix((npp,nvtx)) 
		by = csr_matrix((npp,nvtx)) 
		self.B = csr_matrix((npp,nu)) 
		self.Q = csr_matrix((npp,npp)) 
		self.f = np.zeros((nu,1)) 
		self.g = np.zeros((npp,1)) 

		if (nngpt==4):  
			gpt=1.0e0/np.sqrt(3.0e0) 
			s[0] = -gpt  
			t[0] = -gpt  
			wt[0]=1 

			s[1] =  gpt  
			t[1] = -gpt  
			wt[1]=1 

			s[2] =  gpt  
			t[2] =  gpt  
			wt[2]=1  

			s[3] = -gpt  
			t[3] =  gpt  
			wt[3]=1 

		elif (nngpt==9):
			gpt=np.sqrt(0.6)  
			s[0] = -gpt  
			t[0] = -gpt  
			wt[0]=25/81 

			s[1] =  gpt  
			t[1] = -gpt  
			wt[1]=25/81 

			s[2] =  gpt  
			t[2] =  gpt  
			wt[2]=25/81  

			s[3] = -gpt  
			t[3] =  gpt  
			wt[3]=25/81 

			s[4] =  0.0  
			t[4] = -gpt  
			wt[4]=40/81 

			s[5] =  gpt  
			t[5] =  0.0  
			wt[5]=40/81 

			s[6] =  0.0  
			t[6] =  gpt  
			wt[6]=40/81  

			s[7] = -gpt  
			t[7] =  0.0  
			wt[7]=40/81 

			s[8] =  0.0  
			t[8] =  0.0  
			wt[8]=64/81 

		xl_v = np.zeros(self.mv.shape)
		yl_v = np.zeros(self.mv.shape)
		for ivtx in range(4):
			xl_v[:,ivtx] = x[self.mv[:,ivtx]] 
			yl_v[:,ivtx] = y[self.mv[:,ivtx]]  

			ae = np.zeros((nel,9,9)) 
			re = np.zeros((nel,9,9)) 
			bbxe = np.zeros((nel,9,9)) 
			bbye = np.zeros((nel,9,9)) 
			bxe = np.zeros((nel,4,9)) 
			bye = np.zeros((nel,4,9)) 
			mpe = np.zeros((nel,4,4)) 
			ge = np.zeros((nel,4)) 

		for igpt in range(nngpt):
			sigpt=s[igpt]
			tigpt=t[igpt]
			wght=wt[igpt]

			jac,invjac,phi,dphidx,dphidy = deriv(sigpt,tigpt,xl_v,yl_v)  
			psi,dpsidx,dpsidy = qderiv(sigpt,tigpt,xl_v,yl_v)         
			for j in range(9):
				for i in range(9):
					ae[:,i,[j]]  = ae[:,i,[j]]  + wght*dpsidx[:,[i]]*dpsidx[:,[j]]*invjac 
					ae[:,i,[j]]  = ae[:,i,[j]]  + wght*dpsidy[:,[i]]*dpsidy[:,[j]]*invjac 
					re[:,i,[j]]  = re[:,i,[j]]  + wght*psi[:,[i]]*psi[:,[j]]*jac 
					bbxe[:,i,[j]] = bbxe[:,i,[j]] - wght*psi[:,[i]] *dpsidx[:,[j]]              
					bbye[:,i,[j]] = bbye[:,i,[j]] - wght*psi[:,[i]] *dpsidy[:,[j]]    

				for i in range(4):
					bxe[:,i,[j]] = bxe[:,i,[j]] - wght*phi[:,[i]] * dpsidx[:,[j]] 
					bye[:,i,[j]] = bye[:,i,[j]] - wght*phi[:,[i]] * dpsidy[:,[j]] 

			for j in range(4):
				for i in range(4):
					mpe[:,i,[j]] = mpe[:,i,[j]] + wght*phi[:,[i]] *phi[:,[j]] *jac 

		for krow in range(9):
			nrow=self.mv[:,krow] 	 
			for kcol in range(9):
				ncol=self.mv[:,kcol] 	  
				self.A = self.A + csr_matrix((ae[:,krow,kcol],(nrow,ncol)),shape = (nu,nu)) 
				self.A = self.A + csr_matrix((ae[:,krow,kcol],(nrow+nvtx,ncol+nvtx)),shape = (nu,nu)) 
				self.G = self.G + csr_matrix((re[:,krow,kcol],(nrow,ncol)),shape = (nu,nu)) 
				self.G = self.G + csr_matrix((re[:,krow,kcol],(nrow+nvtx,ncol+nvtx)),shape = (nu,nu)) 
				self.Bx = self.Bx + csr_matrix((bbxe[:,krow,kcol],(nrow,ncol)),shape = (nvtx,nvtx)) 
				self.By = self.By + csr_matrix((bbye[:,krow,kcol],(nrow,ncol)),shape = (nvtx,nvtx)) 

			for kcol in range(4):
				ncol=self.mp[:,kcol] 	  
				bx = bx + csr_matrix((bxe[:,kcol,krow],(ncol,nrow)),shape = (npp,nvtx)) 
				by = by + csr_matrix((bye[:,kcol,krow],(ncol,nrow)),shape = (npp,nvtx)) 

		self.B = hstack((bx,by),format="csr") 

		for krow in range(4):
			nrow=self.mp[:,krow]	 
			for kcol in range(4):
				ncol=self.mp[:,kcol] 	  
				self.Q = self.Q + csr_matrix((mpe[:,krow,kcol],(nrow,ncol)),shape = (npp,npp)) 
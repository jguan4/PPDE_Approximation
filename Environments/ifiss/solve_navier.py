import numpy as np
from utils import *
from scipy.sparse import spdiags, vstack, hstack, csr_matrix
from scipy.sparse.linalg import lsqr,spsolve,gmres
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# def solve_navier(grid, maxit, viscosity, tol_nl):

# 	maxit_p = maxit

# 	[Ast,Bst,fst,gst] = flowbc(A,B,f,g,xy,bound)
# 	nlres0_np.linalg.norm = np.linalg.norm([fstgst])

# 	nv=len(fst)/2
# 	np=len(gst)
	
# 	xstz=[Ast,Bst',np.zeros(2*nv,1)Bst,sparse(np,np),ones(np,1)/np 
# 	np.zeros(1,2*nv),ones(1,np)/np,np.zeros(1,1)]\[fstgst0]
# 	xst=xstz(1:end-1) multiplier=xstz(end)
	
# 	N = navier_q2(xy,mv,xst)

# 	# T = spdiags(diag(Q),0,size(xyp,1),size(xyp,1))
# 	# F = viscosity*A+ + [N, sparse(nv,nv) sparse(nv,nv), N] 
# 	# [Fnst,Bst,fst,gst] = flowbc(F,B,f,g,xy,bound)
# 	# al_factor = Bst'*(T\Bst)
# 	# Anst = Fnst + itslv_params.gamma*al_factor 
# 	# fst = fst + itslv_params.gamma*Bst'*(T\gst)

# 	Anst = viscosity*A + [N, sparse(nv,nv) sparse(nv,nv), N]
# 	[Anst,Bst,fst,gst] = flowbc(Anst,B,f,g,xy,bound)

# 	nlres = [Anst,Bst'Bst,sparse(np,np)]*xst-[fstgst]
# 	nlres_np.linalg.norm  = np.linalg.norm(nlres)

# 	flowsol = xst
# 	flowsols = flowsol

# 	while nlres_np.linalg.norm>nlres0_np.linalg.norm*tol_nl && it_p<maxit_p
# 		it_p = it_p+1
# 		dxnsz= -[Anst,Bst',np.zeros(2*nv,1)Bst,sparse(np,np),ones(np,1)/np ...
# 				np.zeros(1,2*nv),ones(1,np)/np,np.zeros(1,1)]\[nlres0]
# 		dxns=dxnsz(1:end-1) multiplier=dxnsz(end)
# 		xns = flowsol + dxns
# 		N = navier_q2(xy,mv,xns)

# 		# F = viscosity*A+ + [N, sparse(nv,nv) sparse(nv,nv), N] 
# 		# [Fnst,Bst,fst,gst] = flowbc(F,B,f,g,xy,bound)
# 		# al_factor = Bst'*(T\Bst)
# 		# Anst = Fnst + itslv_params.gamma*al_factor 
# 		# fst = fst + itslv_params.gamma*Bst'*(T\gst)

# 		Anst = viscosity*A + [N, sparse(nv,nv) sparse(nv,nv), N]
# 		[Anst,Bst,fst,gst] = flowbc(Anst,B,f,g,xy,bound)

# 		nlres = [Anst,Bst'Bst,sparse(np,np)]*xns-[fstgst]
# 		nlres_np.linalg.norm = np.linalg.norm(nlres)
# 		flowsol = xns
# 		flowsols = np.hstack([flowsols, flowsol])

# 	if nlres_np.linalg.norm <= nlres0_np.linalg.norm * tol_nl, 
# 		print('\nfinished, nonlinear convergence test satisfied\n\n')
# 		nlres = nlres - [np.zeros(2*nv,1)(sum(nlres(2*nv+1:2*nv+np))/np)*ones(np,1)]
# 	else
# 		print('\nfinished, stopped on iteration counts\n\n')

# 	return flowsols

def it_solve_navier(grid, viscosity, max_it, tol_nl, precon, gamma, net = None):

	print('FAST solution of flow problem in square domain ...\n')
	nlmethod=0
	maxit_p = max_it

	itslv_params = {'itmeth':1, 'tol':1e-6, 'maxit':200, 'gamma':gamma, 'precon':0, 'net':net}

	print('stokes system ...\n')
	## boundary conditions
	Ast,Bst,fst,gst = flowbc(grid.A,grid.B,grid.f,grid.g,grid.xy,grid.bound)
	nlres0_norm = np.linalg.norm(np.vstack([fst,gst])) 

	nv=len(fst)//2  
	npp=len(gst) 

	xst = it_nstokes(grid, Ast, Bst, grid.Q, grid.G, np.vstack([fst,gst]), np.zeros((nv*2,1)), viscosity, itslv_params) 
	flowplot(grid, nv, npp, xst)

	itslv_params['precon'] = precon

	# compute residual of Stokes solution
	N = navier_q2(grid.xy,grid.mv,xst) 
	T = spdiags(grid.Q.diagonal(),0,len(grid.xyp),len(grid.xyp),format="csc") 

	F = viscosity*grid.A+ concat_matrix(N,csr_matrix((nv,nv)),csr_matrix((nv,nv)), N)  
	[Fnst,Bst,fst,gst] = flowbc(F,grid.B,grid.f,grid.g,grid.xy,grid.bound) 
	al_factor = Bst.transpose()*(spsolve(T,Bst.tocsc()))
	Anst = Fnst + gamma*al_factor 
	fstadd = gamma*Bst.transpose()*(spsolve(T,gst))
	fst = fst + np.reshape(fstadd, (len(fstadd),1))

	nlres = concat_matrix(Anst,Bst.transpose(),Bst, csr_matrix((npp,npp)))*xst-np.vstack([fst,gst]) 
	nlres_norm  = np.linalg.norm(nlres) 
	
	print('\n\ninitial nonlinear residual is {0}'.format(nlres0_norm))
	print('\nStokes solution residual is {0}\n'.format(nlres_norm))
	flowsol = xst 

	it_p = 0 
	tau_safety = 1e-2 

	# usols = [] 

	# nonlinear iteration
	## Picard startup step
	while nlres_norm>nlres0_norm*tol_nl and it_p<maxit_p:
		it_p = it_p+1 
		print('\nPicard iteration number {0} \n'.format(it_p)),
		# stopping tolerance
		# if exact~=1, itslv_params.tol = tol_nl/10 
		itslv_params['tol'] = np.amax([tau_safety*nlres_norm,tol_nl/10]) 
		# end
		print(flowsol)
		input()

		# compute Picard correction and update solution
		dxns = -it_nstokes(grid, Anst, Bst, T, grid.G, nlres, flowsol, viscosity, itslv_params) 

		xns = flowsol + dxns 
		flowplot(grid, nv, npp, xns)

		# usols = hstack([usols,xns[0:2*nv]]) 
		# compute residual of new solution
		N = navier_q2(grid.xy,grid.mv,xns) 

		F = viscosity*grid.A+ concat_matrix(N,csr_matrix((nv,nv)),csr_matrix((nv,nv)), N)  
		[Fnst,Bst,fst,gst] = flowbc(F,grid.B,grid.f,grid.g,grid.xy,grid.bound) 
		al_factor = Bst.transpose()*(spsolve(T,Bst.tocsc()))
		Anst = Fnst + gamma*al_factor 
		fstadd = gamma*Bst.transpose()*(spsolve(T,gst))
		fst = fst + np.reshape(fstadd, (nv*2,1))

		lhs = concat_matrix(Anst,Bst.transpose(),Bst, csr_matrix((npp,npp)))
		nlres = lhs*xns-np.vstack([fst,gst]) 
		nlres_norm  = np.linalg.norm(nlres) 
		nnv=len(fst)  
		soldiff=np.linalg.norm(xns[0:nnv]-flowsol[0:nnv]) 
		print('nonlinear residual is {0}'.format(nlres_norm))
		print('\n   velocity change is {0}\n'.format(soldiff))
		flowsol = xns 
		# end of Picard iteration loop

	if nlres_norm <= nlres0_norm * tol_nl:
		print('\nfinished, nonlinear convergence test satisfied\n\n') 
	else:
		print('\nfinished, stopped on iteration counts\n\n') 

	xitns=flowsol 

	## estimate errors
	# navierpost
	return it_p


def it_nstokes(grid, Anst, Bst, T, G, nlres, flowsol, viscosity, itslv_params):

	# set parameters for iteration
	tol           = itslv_params['tol'] 
	maxit         = itslv_params['maxit'] 
	precon        = itslv_params['precon']

	### NAVIER-STOKES Problem
	# set structure for preconditioner
	n_null = Bst.shape[0]
	nv1 = Anst.shape[0]//2 
	vflow = flowsol[0:2*nv1] 

	# set structure for matrix-vector product
	afun_par = {'F':Anst,'B':Bst} 
	print('augmented lagrangian preconditioning ...\n')

	if precon == 1:
		mfun_par = {'F':Anst, 'B':Bst, 'T':T, 'viscosity':viscosity,'gamma':itslv_params['gamma'], 'precon':precon} 
	elif precon == 0:
		mfun_par = {'precon':precon}
	elif precon == 2:
		net = itslv_params['net']
		mfun_par = {'net':net, 'flowsol':flowsol}
	else:
		pass
	
	# solve using GMRES or BiCGSTAB(ell) or IDR(s)
	# zero initial guess
	x0=np.zeros(nlres.shape) 

	print('GMRES iteration ... ') 
	params = [tol,maxit,1] 
	x_it,flag,itera,resvec = gmres_r(afun_par,mfun_par,nlres,params,x0) 
	
	# Print and plot results
	if flag==0:
		# successful convergence
		print('convergence in {0} iterations\n\n'.format(itera))
	else:
		print('iteration aborted! Iteration returned with flag equal to  {0} \n'.format(flag))
		print('maximum iteration count of {0} reached\n'.format(maxit)) 
		print('relative residual is {0}\n'.format(resvec[-1]/resvec[0])) 
		### plot residuals
	return x_it


def gmres_r(aparams, mparams, b, params, x0):

	# Names of mvp and preconditioning routines

	n=len(b)
	errtol=params[0]
	kmax=params[1]
	reorth=1

	x=x0
	flag=0

	h=np.zeros((kmax+1,kmax))
	v=np.zeros((n,kmax+1))
	c=np.zeros((kmax+1,1))
	s=np.zeros((kmax+1,1))

	if np.linalg.norm(x) !=0:
		aparams['x'] = x
		r = b-a_al(**aparams)
	else:
		r = b

	rho=np.linalg.norm(r)
	g=rho*np.eye(kmax+1,1)
	errtol=errtol*np.linalg.norm(b)
	error=[]

	# test for termination on entry
	error.append(rho)
	total_iters=0       
	if(rho < errtol):
		return

	v[:,[0]]=r/rho
	beta=rho
	k=-1

	# GMRES iteration
	while ((rho > errtol) and (k < kmax-1)):

		k=k+1 #k=0
		## Modification by HCE, 15 Oct. 2004: adapt for right preconditioning
		mparams['x'] = v[:,[k]] #v0
		if mparams['precon'] == 1:
			mout = m_al(**mparams)
		elif mparams['precon'] == 0:
			mout = v[:,[k]]
		elif mparams['precon'] == 2:
			mout = m_net(**mparams)
		else:
			pass
		aparams['x'] = mout
		v[:,[k+1]] = a_al(**aparams) #v1 = out
		normav=np.linalg.norm(v[:,k+1]) #v1

		# Modified Gram-Schmidt
		for j in range(k+1): #j=0,...,k = 0
			h[j,k]=np.dot(v[:,j],v[:,k+1])           # h(j,k)=v(:,k+1)'*v(:,j)
			v[:,k+1]=v[:,k+1]-h[j,k]*v[:,j]
		
		h[k+1,k]=np.linalg.norm(v[:,k+1])
		normav2=h[k+1,k]

		# Reorthogonalize?
		if  (reorth == 1 and normav + .001*normav2 == normav) or (reorth ==  3):
			for j in range(k+1):
				hr=np.dot(v[:,j],v[:,k+1])                 # hr=v(:,k+1)'*v(:,j)
				h[j,k]=h[j,k]+hr
				v[:,k+1]=v[:,k+1]-hr*v[:,j]
			h[k+1,k]=np.linalg.norm(v[:,k+1])

		#   watch out for happy breakdown 
		if(h[k+1,k] != 0):
			v[:,k+1]=v[:,k+1]/h[k+1,k]

		#   Form and store the information for the new Givens rotation
		if k>0:
			res = givapp(c[0:k],s[0:k],h[0:k+1,k],k)
			h[0:k+1,k]=res

		nu=np.linalg.norm(h[k:k+2,k])
		if nu!=0:
			c[k]=np.conj(h[k,k]/nu)             #   c(k)=h(k,k)/nu  # Change 6/3/97
			s[k]=-h[k+1,k]/nu 
			h[k,k]=c[k]*h[k,k]-s[k]*h[k+1,k]
			h[k+1,k]=0
			g[k:k+2]=givapp(c[k],s[k],g[k:k+2],1)

		# Update the residual np.linalg.norm
		rho=np.abs(g[k+1,0])         #print('  #5i     #8.4f \n',k,log10(rho))
		error.append(rho)

	# At this point either k > kmax or rho < errtol.
	# It's time to compute x and leave.
	#
	# set flag first
	if rho<errtol:
		flag=0 
	else:
		flag=1

	y=np.linalg.solve(h[0:k+1,0:k+1],g[0:k+1])
	total_iters=k+1

	## Modification by HCE, 15 Oct. 2005: adapt for right preconditioning
	x_new = np.matmul(v[0:n,0:k+1],y)
	mparams['x'] = x_new
	if mparams['precon'] == 1:
		mout = m_al(**mparams)
	elif mparams['precon'] == 0:
		mout = mparams['x']
	elif mparams['precon'] == 2:
		mout = m_net(**mparams)
	else:
		pass
		
	x = x0 + mout

	# error = error.transpose()
	return x, flag, total_iters, error

def m_al(x, F, B, T, viscosity, gamma, precon = 1):
	nv = F.shape[0]
	npp = T.shape[0] 
	rv=x[0:nv]  
	rp=x[nv:nv+npp] 
	zp = -(viscosity+gamma)*spsolve(T,rp)
	zp = np.reshape(zp,(npp,1))
	rv = rv-(B.transpose())*zp 
	zv = spsolve(F, rv) 
	zv = np.reshape(zv,(nv,1))
	y = np.vstack([zv,zp]) 
	return y


def a_al(x, F, B):
	nu = F.shape[0]
	npp = B.shape[0]    
	y = np.vstack([F * x[0:nu] + B.transpose() * x[nu:nu+npp], B * x[0:nu]])
	return y

def m_net(net, x, flowsol):
	pass

def givapp(c,s,vin,k):
	vrot=vin
	for i in range(k):
		w1=c[i]*vrot[i]-s[i]*vrot[i+1]        # Change on next line, 6/3/97
		w2=s[i]*vrot[i]+np.conj(c[i])*vrot[i+1]  # w2=s(i)*vrot(i)+c(i)*vrot(i+1);
		vrot[i:i+2]=[w1,w2]
	return vrot

def flowplot(grid, nv, npp, xst):
	ux = xst[0:nv]
	uy = xst[nv:nv*2]
	ux_mesh = np.reshape(ux, [grid.n+1, grid.n+1])
	uy_mesh = np.reshape(uy, [grid.n+1, grid.n+1])
	p = xst[2*nv:2*nv+npp]
	p_mesh = np.reshape(p, [grid.np+1, grid.np+1])
	fig = plt.figure(figsize=plt.figaspect(.5))
	ax = fig.add_subplot(1, 2, 1)
	ax.streamplot(grid.X, grid.Y, ux_mesh, uy_mesh, cmap=cm.coolwarm)
	ax = fig.add_subplot(1, 2, 2, projection='3d')
	# Plot the surface.
	surf = ax.plot_surface(grid.Xp, grid.Yp, p_mesh, cmap=cm.coolwarm, linewidth=0, antialiased=False)

	# Add a color bar which maps values to colors.
	fig.colorbar(surf, shrink=0.5, aspect=5)
	plt.show()
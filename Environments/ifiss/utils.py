import numpy as np
from scipy.sparse import spdiags, vstack, hstack, csr_matrix

def deriv(s,t,xl,yl):
	nel = len(xl[:,0])
	zero_v = np.zeros((nel,1))
	one_v = np.ones((nel,1))

	phi_e, dphids, dphidt = shape(s,t)
	dxds = np.zeros((nel,1))
	dxdt = np.zeros((nel,1))
	dyds = np.zeros((nel,1))
	dydt = np.zeros((nel,1))
	jac = np.zeros((nel,1))
	invjac = np.zeros((nel,1))

	for ivtx in range(4):
		dxds = dxds + xl[:,[ivtx]] * one_v*dphids[ivtx]
		dxdt = dxdt + xl[:,[ivtx]] * one_v*dphidt[ivtx]
		dyds = dyds + yl[:,[ivtx]] * one_v*dphids[ivtx]
		dydt = dydt+ yl[:,[ivtx]] * one_v*dphidt[ivtx]

	jac = dxds*dydt - dxdt*dyds
	invjac = one_v / jac 

	phi = np.zeros((nel,4))
	dphidx = np.zeros((nel,4))
	dphidy = np.zeros((nel,4))
	for ivtx in range(4):
		phi[:,[ivtx]] = phi_e[ivtx]*one_v 
		dphidx[:,[ivtx]] = ( dphids[ivtx]*dydt - dphidt[ivtx]*dyds)
		dphidy[:,[ivtx]] = (-dphids[ivtx]*dxdt + dphidt[ivtx]*dxds)

	return jac, invjac, phi, dphidx, dphidy

def qderiv(s,t,xl,yl):
	nel=len(xl[:,0]) 
	zero_v = np.zeros((nel,1)) 
	one_v = np.ones((nel,1)) 

	phi_e,dphids,dphidt = shape(s,t) 
	psi_e,dpsids,dpsidt = qshape(s,t) 

	dxds = np.zeros((nel,1))  
	dxdt = np.zeros((nel,1))  
	dyds = np.zeros((nel,1))  
	dydt = np.zeros((nel,1))  
	jac = np.zeros((nel,1))  
	invjac = np.zeros((nel,1))   

	for ivtx in range(4):
		dxds = dxds + xl[:,[ivtx]] * one_v*dphids[ivtx] 
		dxdt = dxdt + xl[:,[ivtx]] * one_v*dphidt[ivtx] 
		dyds = dyds + yl[:,[ivtx]] * one_v*dphids[ivtx] 
		dydt = dydt + yl[:,[ivtx]] * one_v*dphidt[ivtx] 

	jac = dxds*dydt - dxdt*dyds 
	invjac = one_v / jac 

	psi = np.zeros((nel,9))
	dpsidx = np.zeros((nel,9))
	dpsidy = np.zeros((nel,9))
	for ivtx in range(9):
		psi[:,[ivtx]] = psi_e[ivtx]*one_v 
		dpsidx[:,[ivtx]] = ( dpsids[ivtx]*dydt - dpsidt[ivtx]*dyds) 
		dpsidy[:,[ivtx]] = (-dpsids[ivtx]*dxdt + dpsidt[ivtx]*dxds) 
	return psi, dpsidx, dpsidy

def shape(s,t):
	one = 1.0e0
	phi = np.zeros(4)
	phi[0] = 0.25 * (s-one)*(t-one)
	phi[1] = -0.25 * (s+one)*(t-one)
	phi[2] = 0.25 * (s+one)*(t+one)
	phi[3] = -0.25 * (s-one)*(t+one)

	dphids = np.zeros(4)
	dphids[0] = 0.25 * (t-one)
	dphids[1] = -0.25 * (t-one)
	dphids[2] = 0.25 * (t+one)
	dphids[3] = -0.25 * (t+one)

	dphidt = np.zeros(4)
	dphidt[0] = 0.25 * (s-one)
	dphidt[1] = -0.25 * (s+one)
	dphidt[2] = 0.25 * (s+one)
	dphidt[3] = -0.25 * (s-one)

	return phi, dphids, dphidt

def qshape(s,t):
	one = 1.0e0 

	ellx = np.zeros((3))
	ellx[0] = 0.5*s*(s-1)   
	ellx[1] = 1-(s*s)       
	ellx[2] = 0.5*s*(s+1)   
	
	elly = np.zeros((3))
	elly[0] = 0.5*t*(t-1) 
	elly[1] = 1-(t*t) 
	elly[2] = 0.5*t*(t+1) 

	dellx = np.zeros((3))
	dellx[0] = s-0.5        
	dellx[1] = -2*s         
	dellx[2] = s+0.5        
	
	delly = np.zeros((3))
	delly[0] = t-0.5 
	delly[1] = -2*t 
	delly[2] = t+0.5 

	psi = np.zeros(9)
	dpsids = np.zeros(9)
	dpsidt = np.zeros(9)
	psi[0] = ellx[0]*elly[0] 
	psi[1] = ellx[2]*elly[0] 
	psi[2] = ellx[2]*elly[2] 
	psi[3] = ellx[0]*elly[2] 
	psi[4] = ellx[1]*elly[0] 
	psi[5] = ellx[2]*elly[1] 
	psi[6] = ellx[1]*elly[2] 
	psi[7] = ellx[0]*elly[1] 
	psi[8] = ellx[1]*elly[1] 
	dpsids[0] = dellx[0]*elly[0] 
	dpsids[1] = dellx[2]*elly[0] 
	dpsids[2] = dellx[2]*elly[2] 
	dpsids[3] = dellx[0]*elly[2] 
	dpsids[4] = dellx[1]*elly[0] 
	dpsids[5] = dellx[2]*elly[1] 
	dpsids[6] = dellx[1]*elly[2] 
	dpsids[7] = dellx[0]*elly[1] 
	dpsids[8] = dellx[1]*elly[1] 
	dpsidt[0] = ellx[0]*delly[0] 
	dpsidt[1] = ellx[2]*delly[0] 
	dpsidt[2] = ellx[2]*delly[2] 
	dpsidt[3] = ellx[0]*delly[2] 
	dpsidt[4] = ellx[1]*delly[0] 
	dpsidt[5] = ellx[2]*delly[1] 
	dpsidt[6] = ellx[1]*delly[2] 
	dpsidt[7] = ellx[0]*delly[1] 
	dpsidt[8] = ellx[1]*delly[1] 

	return psi, dpsids, dpsidt

def navier_q2(xy,mv,flowsol):
	tout = 1
	nngpt=9 
	x=xy[:,0] 
	y=xy[:,1]
	nvtx=len(x)   
	nel=len(mv[:,0])
	usol=flowsol[0:nvtx] 
	vsol=flowsol[nvtx:2*nvtx] 

	# if tout==1:
		# print('setting up Q2 convection matrix...  ')

	#
	# initialise global matrices
	n = csr_matrix((nvtx,nvtx))

	s = np.zeros(nngpt)
	t = np.zeros(nngpt)
	wt = np.zeros(nngpt)
	# set up 3x3 Gauss points
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
	
	xl_v = np.zeros((mv.shape[0], 4))
	yl_v = np.zeros((mv.shape[0], 4))
	# inner loop over elements    
	for ivtx in range(4):
		xl_v[:,ivtx] = x[mv[:,ivtx]]
		yl_v[:,ivtx] = y[mv[:,ivtx]]

	xsl = np.zeros((mv.shape[0], 9))
	ysl = np.zeros((mv.shape[0], 9))
	for idx in range(9):		
		xsl[:,[idx]] = usol[mv[:,idx]]
		ysl[:,[idx]] = vsol[mv[:,idx]]

	ne = np.zeros((nel,9,9))
	# 
	# loop over Gauss points
	for igpt in range(nngpt):
		sigpt=s[igpt]
		tigpt=t[igpt]
		wght=wt[igpt]

		#  evaluate derivatives etc
		jac,invjac,phi,dphidx,dphidy = deriv(sigpt,tigpt,xl_v,yl_v)
		psi,dpsidx,dpsidy = qderiv(sigpt,tigpt,xl_v,yl_v)

		u_x = np.zeros((nel,1)) 
		u_y=np.zeros((nel,1))
		for k in range(9):
			u_x = u_x + xsl[:,[k]] * psi[:,[k]]
			u_y = u_y + ysl[:,[k]] * psi[:,[k]]
		
		for j in range(9):
			for i in range(9):
				ne[:,i,[j]]  = ne[:,i,[j]]  + wght*u_x*psi[:,[i]]*dpsidx[:,[j]]
				ne[:,i,[j]]  = ne[:,i,[j]]  + wght*u_y*psi[:,[i]]*dpsidy[:,[j]]
		# end of Gauss point loop

	##  element assembly into global matrix
	for krow in range(9):
		nrow=mv[:,krow]
		for kcol in range(9):
			ncol=mv[:,kcol]
			n = n + csr_matrix((ne[:,krow,kcol],(nrow,ncol)),shape = (nvtx,nvtx))

	# print('done.\n')
	return n

def specific_flow(xbd,ybd):
	bcx=0*xbd
	bcy=0*xbd
	k=np.argwhere((ybd==1) & (xbd>-1) & (xbd<1)) 
	bcx[k]=(1-xbd[k]*xbd[k])*(1+xbd[k]*xbd[k]);
	return bcx, bcy

def flowbc(A,B,f,g,xy,bound):
	nu = len(f)
	npp = len(g)

	nvtx = nu//2
	nbd=len(bound)

	null_col=csr_matrix((nvtx,nbd))
	null_pcol=csr_matrix((npp,nbd))

	Ax=A[0:nvtx,0:nvtx]
	Ay=A[nvtx:nu,nvtx:nu]
	Bx=B[0:npp,0:nvtx]
	By=B[0:npp,nvtx:nu]
	gz=g

	xbd=xy[bound,0]
	ybd=xy[bound,1]

	bcx,bcy=specific_flow(xbd,ybd)
	bounds = np.vstack([bound,bound+nvtx])

	f1 = f - A[:,bounds.flatten()]*np.vstack([bcx,bcy])
	fx = f1[0:nvtx]
	fy=f1[nvtx:nu]
	gz = gz - Bx[:,bound.flatten()]*bcx
	gz = gz - By[:,bound.flatten()]*bcy

	dA=np.zeros((nvtx))
	dA[bound.flatten()]=np.ones((nbd))

	Axt = Ax.transpose()
	Axt[:,bound.flatten()] = null_col
	Ax = Axt.transpose()
	Ax[:,bound.flatten()] = null_col
	Ax=Ax+spdiags(dA,0,nvtx,nvtx)
	fx[bound.flatten()]=bcx

	Ayt = Ay.transpose()
	Ayt[:,bound.flatten()] = null_col
	Ay = Ayt.transpose()
	Ay[:,bound.flatten()] = null_col
	Ay=Ay+spdiags(dA,0,nvtx,nvtx)
	fy[bound.flatten()]=bcy

	Bx[:,bound.flatten()]=null_pcol
	By[:,bound.flatten()]=null_pcol

	az=concat_matrix(Ax,csr_matrix((nvtx,nvtx)),csr_matrix((nvtx,nvtx)),Ay)

	bz=hstack([Bx,By],format="csr")
	fz=np.vstack([fx,fy])

	return az, bz, fz, gz

def concat_matrix(A,B,C,D):
	# build matrix of the form [A,B;C,D]
	return vstack([hstack([A,B],format="csr"),hstack([C,D],format="csr")],format="csr")
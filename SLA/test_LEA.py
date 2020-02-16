import numpy as np
from LEA import LEA

# learning steps
n=10
# number of models
N=3
# number of locations
ltln=2

# define artificial predictions
f = np.zeros((n,N,ltln))

f[:,0,:] =1
f[:,1,:] =0.2
f[:,2,:] =0

# define artificial observations
y = np.ones((n,ltln))*0.7

# define maximal range (in EGA it is done inside the function
# in LEA it is done outside)
M=np.max(np.max(abs(2*(f-y[:,None])*f),0),0)

# number of learning rates
neta=10
# Chosen here to increase exponentially
eta=256.*2.**(-np.arange(neta,0,-1))  
# The learning rate of eta ("the eta of eta")
leta=1
# model weight (different for each eta)
w = np.zeros((n+1,neta,N,ltln))
# eta weight
ew = np.zeros((n+1,neta,ltln))

# define initial model and eta weight
w[0,:]=1./N
ew[0,:]=1./neta

'''
w=w[0,:,:,:]
ew=ew[0,:,:]
f=f[0,:,:]
y=y[0,:]
'''

# run the LEA algorithm to extract weights
for t in range(1,n+1):
  ew[t,...],w[t,...]=LEA(f[t-1,...],y[t-1,...],w[t-1,...],ew[t-1,...],eta,leta,M)

# total model weight
mw=np.sum(ew[...,None,:]*w,1)

# hindcast
F=np.sum(mw[0:-1,:]*f,axis=1)
# last time step of w is the weight for the next time step for forecast

print(mw)

#plot
import matplotlib 
import numpy as np 
import matplotlib.pyplot as plt

fig, (ax0,ax1) = plt.subplots(2)

t = np.arange(1, w.shape[0]+1, 1)

#plot weights of the models
ax0.set_ylabel('Modle weights')
ax0.set_xlabel('time')
ax0.plot(t,mw[:,0,0], 'r', label='E1')
ax0.plot(t,mw[:,1,0], 'g', label='E2')
ax0.plot(t,mw[:,2,0], 'b', label='E3')
ax0.legend(loc="lower right") 

ax1.set_ylabel('Forecast and observations')
ax1.set_xlabel('time')
ax1.plot(t[0:-1], f[:,0,0], 'r', label='E1')
ax1.plot(t[0:-1], f[:,1,0], 'g', label='E2')
ax1.plot(t[0:-1], f[:,2,0], 'b', label='E3')
ax1.plot(t[0:-1], F[:,0]     , 'm', label='F')
ax1.plot(t[0:-1], y[:,0]    , 'k:', label='Y')
ax1.legend(loc="lower right") 


plt.subplots_adjust(hspace=0.3)
plt.show()
	
	
	
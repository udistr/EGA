import numpy as np
from SLA import LEA

# learning steps
n=10
# number of models
N=3
# number of locations
ltln=2

# define artificial predictions
X = np.zeros((n,N,ltln))

X[:,0,:] =1
X[:,1,:] =0.2
X[:,2,:] =0

# define artificial observations
y = np.ones((n,ltln))*0.7

A=LEA()
w1=A.fit(X,y)
w=A.mw

# hindcast
F=np.sum(w[0:-1,:]*X,axis=1)
# last time step of w is the weight for the next time step for forecast

#print(w)

B=LEA()
w2=B.fit(X[0:3,:],y[0:3])
w3=B.update(X[0,:],y[0])


def test_lea_update():
  assert np.all(w3==w[4,:])



'''
#plot
import matplotlib 
import numpy as np 
import matplotlib.pyplot as plt

fig, (ax0,ax1) = plt.subplots(2)

t = np.arange(1, w.shape[0]+1, 1)

#plot weights of the models
ax0.set_ylabel('Modle weights')
ax0.set_xlabel('time')
ax0.plot(t,w[:,0,0], 'r', label='E1')
ax0.plot(t,w[:,1,0], 'g', label='E2')
ax0.plot(t,w[:,2,0], 'b', label='E3')
ax0.legend(loc="lower right") 

ax1.set_ylabel('Forecast and observations')
ax1.set_xlabel('time')
ax1.plot(t[0:-1], X[:,0,0], 'r', label='E1')
ax1.plot(t[0:-1], X[:,1,0], 'g', label='E2')
ax1.plot(t[0:-1], X[:,2,0], 'b', label='E3')
ax1.plot(t[0:-1], F[:,0]     , 'm', label='F')
ax1.plot(t[0:-1], y[:,0]    , 'k:', label='Y')
ax1.legend(loc="lower right") 


plt.subplots_adjust(hspace=0.3)
plt.show()
	
'''	

# learning steps
n=10
# number of models
N=3

# define artificial predictions
X = np.zeros((n,N,ltln))

X[:,0] =1
X[:,1] =0.2
X[:,2] =0

# define artificial observations
y = np.ones((n,ltln))*0.7

A=LEA()
w1=A.fit(X,y)
w=A.mw

# hindcast
F=np.sum(w[0:-1,:]*X,axis=1)
# last time step of w is the weight for the next time step for forecast

#print(w)

B=LEA()
w2=B.fit(X[0:3,:],y[0:3])
w3=B.update(X[0,:],y[0])

def test_lea2D_update():
  assert np.all(w3==w[4,:])
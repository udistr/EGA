import numpy as np
from SLA import EGA

# learning steps
n=10
# number of models
N=3

# define artificial predictions
f = np.zeros((n,N))

f[:,0] =1
f[:,1] =0.7
f[:,2] =0

# define artificial observations
y = np.ones((n))*0.7

# run the EGA algorithm to extract weights
w=EGA(f,y)

# hindcast
F=np.sum(w[0:-1,:,0]*f,axis=1)
# last time step of w is the weight for the next time step

print(w)

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
ax1.plot(t[0:-1], f[:,0], 'r', label='E1')
ax1.plot(t[0:-1], f[:,1], 'g', label='E2')
ax1.plot(t[0:-1], f[:,2], 'b', label='E3')
ax1.plot(t[0:-1], F     , 'm', label='F')
ax1.plot(t[0:-1], y     , 'k:', label='Y')
ax1.legend(loc="lower right") 


plt.subplots_adjust(hspace=0.3)
plt.show()
	
	
	
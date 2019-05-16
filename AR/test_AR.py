import numpy as np
from AR import AR

n=10
N=3

f = np.zeros((n,N))

f[:,0] =1+np.random.uniform(-0.1,0.1,n)
f[:,1] =0.7+np.random.uniform(-0.1,0.1,n)
f[:,2] =0+np.random.uniform(-0.1,0.1,n)
w=1/N

p=np.sum(w*f,axis=1)
P=p[..., np.newaxis]
sigma=np.sqrt(np.sum(w*(f-P)**2,axis=1))

y = np.ones((n))*0.7+np.random.uniform(-0.2,0.2,n)
treshold=0.0

g=np.zeros((2,11))

for i in np.arange(0,11,1):
  g[:,i]=AR(f,w,y,0.1*i)

print(g)	
	
import matplotlib 
import numpy as np 
import matplotlib.pyplot as plt

fig, (ax1,ax2) = plt.subplots(2)

t = np.arange(1, y.shape[0]+2, 1)

ax1.set_ylabel('Forecast and observations')
ax1.set_xlabel('time')
ax1.plot(t[0:-1], y     , 'k', label='Y')
ax1.plot(t[0:-1], f[:,0], 'r', label='E1')
ax1.plot(t[0:-1], f[:,1], 'g', label='E2')
ax1.plot(t[0:-1], f[:,2], 'b', label='E3')
ax1.plot(t[0:-1], p, 'm', label='P')
ax1.legend(bbox_to_anchor=(0,1), loc="upper left") 

ax2.plot(t[0:-1], y     , 'k', label='Y')
ax2.plot(t[0:-1], p-g[1,1]*sigma, 'b-.', label=r'$P-\gamma_d \cdot \sigma$')
ax2.plot(t[0:-1], p, 'b', label='P')
ax2.plot(t[0:-1], p+g[0,1]*sigma, 'b--', label=r'$P+\gamma_u \cdot \sigma$')
ax2.legend(bbox_to_anchor=(0,1), loc="upper left") 
ax1.set_ylabel('AR calibration')
ax1.set_xlabel('time')

plt.subplots_adjust(hspace=0.3)
plt.show()





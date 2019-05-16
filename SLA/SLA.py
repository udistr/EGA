import numpy as np
import sys


def loss_ega(f,y,p):
 l=2*(p-y)*f
 return(l)

def exp_grad_weight(f,y,e):
  """The weight of the Exponential Gradient Average (EGA) Forecaster

  Parameters
  ----------
  f : array_like
      The ensemble forecast. For N models and n time steps and ltln locations f size is f(n,N,ltln).
  y : array_like
      The observations y(n,ltln). 
  e : scalar
      learning rate
 Returns
 -------
  out: weights
       array of the size (n+1,N,ltln). The last time is the weight for the future.
  """
  dim=f.shape
  lrt=dim[0]
  N=dim[1]
  ltln=int(np.prod(np.array(dim))/lrt/N)
  
  #************************
  # defining weights array
  #************************
  w=np.zeros(f.shape, dtype='Float64')

  #************************
  #bound for the loss 
  #************************
  Y=y[..., np.newaxis]
  M=np.amax(np.abs(loss_ega(f,Y,f)),axis=(0,1))

  #************************************
  #eta and starting weight definitions  
  #************************************
  eta=e
  w_start=np.ones(f.shape[1:], dtype='Float64')/N

  #************************************
  #first time step weight 
  #************************************
  p = np.ones((lrt+1,ltln), dtype='Float64')
  p[0,:]=np.sum(w_start*f[0,:],axis=0)
  los=loss_ega(f[0,:],Y[0,:],p[0,:])/M
  w[0,:]=w_start*np.exp(-eta*los)/np.sum(w_start*np.exp(-eta*los),axis=0)
  # equation 3 in strobach and bel (2016)

  #************************************
  #Updating the weights 
  #************************************
  for t in range(1,lrt):
    p[t,:]=np.sum(w[t-1,:]*f[t,:],axis=0)
    los=(loss_ega(f[t,:],Y[t,:],p[t,:]))/M
    w[t,:]=w[t-1,:]*np.exp(-eta*los)/np.sum(w[t-1,:]*np.exp(-eta*los),axis=0)
    # equation 3 in strobach and bel (2016)
    # avoid 0 weight that may be caused due to computer accuracy
    w[t,:]=np.clip(w[t,:],1e-6,None) 
    w[t,:]=w[t,:]/np.sum(w[t,:],axis=0)

  #************************************
  #Preparing the output 
  #************************************
  w1=np.append(w_start[np.newaxis,...],w,axis=0)
  return w1


def EGA(f,y):
  """The weight of the Exponential Gradient Average (EGA) Forecaster with optimization for the learning rate.

  Parameters
  ----------
  f : array_like
      The ensemble forecast. For N models and n time steps and ltln locations f size is f(n,N,ltln).
  y : array_like
      The observations y(n,ltln). 
      
 Returns
 -------
  out: weights
       array of the size (n+1,N,ltln). The last time is the weight for the future.
  """
  #************************
  # determining dimentions
  #************************

  dim=f.shape
  lrt=dim[0]
  N=dim[1]
  ltln=int(np.prod(np.array(dim))/lrt/N)
  f1=f.reshape((lrt,N,ltln))
  y1=y.reshape((lrt,ltln))


  #*************************************************************
  # Defining variables
  #*************************************************************

  # vector that contains the best eta for each grid cell 
  # and updates after every iteration of the resolution
  start=np.zeros(ltln, dtype='Float64') 
  # index of minimum metric value
  mind=np.zeros(ltln, dtype=int)
  # optimal learning parameter
  eta1=np.zeros(ltln, dtype='Float64')
  # predictions of the algorithm
  expf1=np.zeros((lrt,ltln), dtype='Float64')
  # weights of the models
  weight1=np.zeros((lrt+1,N,ltln), dtype='Float64')
  
  # first iteration
  start=0. # at the beginning eta=0 everywhere
  n=69 # we increase eta n times
  deta=10. # each time by deta

  for res in range(0,4): # 4 iterations
    # array that contains the metric for each eta
    metric=np.zeros((ltln,n), dtype='Float64')
    # n etas
    eta0=np.zeros((ltln,n), dtype='Float64')
    # n predictions for each eta
    expf0=np.zeros((lrt,ltln,n), dtype='Float64')
    # n weights for each eta and model
    weight0=np.zeros((lrt+1,N,ltln,n), dtype='Float64')

    for lp in range(0,n): # predictions for each eta
      eta0[:,lp]=start+lp*deta # adding lp*deta every loop

      #*************************************************************
      # Running the algorithm for a given eta
      #*************************************************************
      temp_weight=exp_grad_weight(f1,y1,eta0[:,lp])

      #output weight for a given eta
      weight0[0:lrt+1,:,:,lp]=temp_weight
      #output forecast for a given eta
      expf0[0:lrt,:,lp]=np.sum(weight0[0:lrt,:,:,lp]*f1[0:lrt,:,:],axis=1) 

      rmse=np.sqrt(np.mean((expf0[0:lrt,:,lp]-y1[0:lrt,:])**2,axis=0))
      if ((lp==0) and (res==0)):
        rmse0=np.sqrt(np.mean((np.mean(f1[0:lrt,:,:],axis=1)-y1[0:lrt,:])**2,axis=0))

      rmse=rmse/rmse0 # RMSE relative to the RMSE of simple average

      # calculation of the stability creteria for the eta optimization metric
      # see strobach and bel (2016) equation 6, the term inside the square brackets.
      stab=np.floor(np.max(np.max(np.abs(weight0[1:lrt+1,:,:,lp]-weight0[0:lrt,:,:,lp]),axis=0),axis=0)*N*5)
      #calculation of the matric
      metric[:,lp]=np.round(rmse*(1+stab)*1e5)/1e5
      metric[:,lp]=np.where(np.isnan(metric[:,lp]),float("inf"),metric[:,lp])
      
    for i in range(0,ltln):
      mind[i]=np.argmin(metric[i,:]) # index of minimum metric value
      eta1[i]=eta0[i,mind[i]] # choosing best eta
      expf1[:,i]=expf0[:,i,mind[i]] # predictions of the best eta
      weight1[:,:,i]=weight0[:,:,i,mind[i]] # weight of best eta

    deta=0.1*deta # increasing eta resolution for the next iteration
    start=np.clip(eta1-9*deta,deta,None) # defining new start points for the eta scanning
    n=19 # running between eta-9*deta to eta+9*deta - 19 scans

    # cleanning variables for the next iteration
    del expf0
    del weight0
    del metric
    del eta0 

  return(weight1)




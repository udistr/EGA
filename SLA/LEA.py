import numpy as np
import sys

def LEA(f,y,w,ew,eta,leta,M):
  """The weight of the Learn Eta Algorithm (LEA) Forecaster
  Parameters
  ----------
  f    : array_like
         The ensemble forecast. For N models and n time steps and ltln locations f size is f(n,N,ltln).
  y    : array_like
         The observations y(n,ltln). 
  w    : array_like
         Initial weight for N models and neta eta's w(N,eta,ltln).
  ew   : array_like
         Initital weight for the eta's ew(N,ltln). 
  eta  : array_like
         learning rate for range of neta eta's eta(neta,ltln)
  leta : scalar
         learning rate of eta
  M    : scalar
         Normalization factor (scales loss between -1 and 1)
 -------
  out: updated weights
  w1   : array of the size (N,eta,ltln). The models' weight (for each eta)
  ew1  : array of the size (N,ltln). Eta's weight
  """
  # calculate total model weight
  mw=np.sum(ew[:,None]*w,0) 
  # calculate forecast
  p=np.sum(mw*f,0)
  # calculate scaled loss (EGA)
  loss=(p-y)*f/M
  # calculate loss per eta
  losspereta=-np.log(np.sum(w*np.exp(-loss),1))
  # update eta's weights
  ew1=ew*np.exp(-leta*losspereta)
  # normalization
  ew1=ew1/(np.sum(ew1,0))
  # update model weight for each eta
  print(w.shape)
  print(eta[:,None,None,...].shape)
  print(loss[None,:,...].shape)
  w1=w*np.exp(-eta[:,None,None,...]*loss[None,:,...])
  # scaling for model weight (sum of weights =1)
  N=(np.sum(w1,1))
  # scale model weight
  w1=w1/N[:,None,...]

  return ew1,w1
  
  
  
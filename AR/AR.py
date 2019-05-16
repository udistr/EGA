import numpy as np

def heaviside(x):
  """heaviside functions

  Parameters
  ----------
  x : array_like
      values

  Returns
  -------
  out: same as x
      1 if x>0 zero otherwise
  """
  x=np.where(x>0,1,0)
  return(x)

def AR(f,w,y,threshold):
  """heaviside functions

  Parameters
  ----------
  f : array_like
      The ensemble forecast. For N models and n time steps f size is f(n,N).
  w : array_like
      The ensemble weights. should be the sane size as f or a constant.
  y : array_like
      The observations y(n). 
  threshold : scalar
      The confidence level (probability to be outside range)
 Returns
 -------
  out: gamma up and gamm down which is used to calculate the range.
  """
  p=np.sum(w*f,axis=1)
  P=p[..., np.newaxis]
  Y=y[..., np.newaxis]

  sigma=np.sqrt(np.sum(w*(f-P)**2,axis=1))
  SIGMA=sigma[..., np.newaxis]
  rmse=np.sqrt(np.mean((p-y)**2))

  lim=np.max(np.abs(p-y)/sigma)
  delta=10**np.floor(np.log10(lim))*10**-2
  n0=int(np.ceil(lim/delta))
  lrt=y.shape[0]
  g=np.tile(np.arange(delta,n0*delta+delta,delta),(lrt,1))

  #fraction of observations inside
  pru=np.mean(heaviside((P+g*SIGMA)-Y),0)
  prd=np.mean(heaviside(Y-(P-g*SIGMA)),0)

  iu=np.argmax(pru>=(1-threshold/2))
  id=np.argmax(prd>=(1-threshold/2))

  #cal(0) is the upper limit gu
  #cal(1) is the lower limit gd
  cal=[g[0,iu],g[0,id]]
  return(cal)


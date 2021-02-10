import numpy as np

class LEA(object):

  def __init__(self,eta=None,w=None,ew=None,leta=1):
    self.eta=eta
    self.w=w
    self.ew=ew
    self.leta=leta

  def list_params(self):
    print("eta")
    print("w")
    print("ew")
    print("leta")

  def print_params(self):
    print("eta = "+str(self.eta))
    print("w = "+str(self.w))
    print("ew = "+str(self.ew))
    print("leta = "+str(self.leta))

  def LEA(self,X,y):
    """The weight of the Learn Eta Algorithm (LEA) Forecaster
    Parameters
    ----------
    X    : array_like
          The ensemble forecast. For N models and n time steps and ltln locations f size is X(n,N,ltln).
    y    : array_like
          The observations y(n,ltln). 
    w    : array_like
          Initial weight for N models and neta eta's w(N,eta,ltln).
    w0   : array_like
          Initital weight for the eta's ew(N,ltln). 
    eta  : array_like
          learning rate for range of neta eta's eta(neta,ltln)
    leta : scalar
          learning rate of eta
    -------
    out: updated weights
    w1   : array of the size (N,eta,ltln). The models' weight (for each eta)
    ew1  : array of the size (N,ltln). Eta's weight
    """

    M=self.M
    leta=self.leta
    eta=self.eta
    ew=self.ew[-1,...]
    w=self.w[-1,...]
    dout=w.shape
    # calculate total model weight
    mw=np.sum(ew[None,...]*w,1) 
    # calculate forecast
    p=np.sum(mw*X,0)
    # calculate scaled loss (EGA)
    loss=(p-y)*X/M
    # calculate loss per eta
    losspereta=-np.log(np.sum(w*np.exp(-loss[:,None,...]),0))
    # update eta's weights
    ew1=ew*np.exp(-leta*losspereta)
    ew1=np.clip(ew1,1e-6,None) 
    # normalization
    ew1=ew1/(np.sum(ew1,0))
    # update model weight for each eta
    reta=np.reshape(eta,(1,)+eta.shape+tuple((np.ones(len(dout)-2).astype(int))))
    w1=w*np.exp(-reta*loss[:,None,...])
    w1=np.clip(w1,1e-6,None) 
    # scaling for model weight (sum of weights =1)
    N=(np.sum(w1,0))
    # scale model weight
    w1=w1/N[None,...]

    return ew1,w1
        

  def fit(self,X,y):

    # define maximal range (in EGA it is done inside the function
    # in LEA it is done outside)ֿ
    if hasattr(self,"M"):
      return print("initial state is defined, use update")
    else:
      M=2*np.amax(np.abs(X-y[:,None]),axis=(0,1))*np.amax(np.abs(X),axis=(0,1))
      self.M=M
    # Chosen here to increase exponentially
    if self.eta is None:
      # number of learning rates
      neta=10
      eta=2.**(-np.arange(neta,0,-1)+8)
    else:
      eta=self.eta
    self.eta=eta
    dim=X.shape
    n=dim[0]
    N=dim[1]
    ltln=int(np.prod(np.array(dim))/n/N)
    dout1 = (dim[1],) + (eta.shape[0],) + dim[2:]
    dout2 = (eta.shape[0],) + dim[2:]
    # model weight (different for each eta)
    w = np.zeros(dout1)
    # eta weight
    ew = np.ones(dout2)

    # define initial model and eta weight
    if self.w is None:
      w=np.ones(w.shape, dtype='Float64')/N
      self.w=w[np.newaxis,...]

    if self.ew is None:
      ew=ew/eta.shape[0]
      self.ew=ew[np.newaxis,...]
    
    self.mw=np.sum(ew[None,...]*w,1)[np.newaxis,...]

    # run the LEA algorithm to extract weights
    for t in range(1,n+1):
      ew,w=self.LEA(X[t-1,...],y[t-1,...])
      self.w=np.append(self.w,w[np.newaxis,...],axis=0)
      self.ew=np.append(self.ew,ew[np.newaxis,...],axis=0)
      # total model weight
      mw=np.sum(ew[None,...]*w,1)
      self.mw=np.append(self.mw,mw[np.newaxis,...],axis=0)

    return mw

  def update(self,X,y):
    """update the weight after first intialization with data (after performing fit). 
    By providing new observations the alogorithm will return updated weight.
    ----------
    X : array_like
        The ensemble forecast. For N models and ltln locations f size is f(N,ltln).
    y : array_like
        The observations y(ltln).        
    Returns
    -------
    out: weights
        array of the size (N,ltln).
    """
    assert (X.shape) == (self.mw[-1].shape)
    assert any(self.M)!=None
    ew,w=self.LEA(X,y)
    mw=np.sum(ew[None,...]*w,1)
    return mw
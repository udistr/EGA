import numpy as np

class EGA(object):
  """The Exponential Gradient Average (EGA) Forecaster
  Parameters
  ----------
  e   : float, default=0.5
      The learning rate
  w0  : int, array_like, default is equal weith (one over the number of forecasters)
  start : float, default=0
      The scan beginning at the outer loop
  nscan : integer, default=69
      number of scans at the outer loop
  delta_eta : float, default=10
      scan delta at the outer loop
  iters : int, default=4
      Number of iteration loops
  save_w : int, default=1
      Whether to save the historical weights
  Attributes
  ----------
  w : ndarray of shape (time,models,samples...)
      The weight history (id save_w=1)
  e   : float, n_samples array of optimized eta (if not provided)

  Examples
  --------
  >>> import numpy as np
  >>> from SLA import EGA

  >>> # learning steps
  >>> n=10
  >>> # number of models
  >>> N=3
  >>> # define artificial predictions
  >>> X = np.zeros((n,N))
  >>> X[:,0] =1
  >>> X[:,1] =0.7
  >>> X[:,2] =0
  >>> # define artificial observations
  >>> y = np.ones((n))*0.7
  >>> # run the EGA algorithm to extract weights
  >>> w=EGA().fit(X,y)
  >>> A.fit(X,y)
  array([0.45289346, 0.35186123, 0.19524531])

  References
  ----------
Strobach, E., and G. Bel., 2017. “Qunatifying the uncertainties in an ensemble
 of decadal climate predictions”. J. Geophys. Res., 122, doi:10.1002/2017JD027249.
Strobach, E. and G. Bel., 2017. “The relative contribution of the internal and
 model variabilities to the uncertainty in decadal climate predictions”.
  Climate dynamics, 1-15, doi:10.1007/s00382-016-3507-7.
Strobach, E., and G. Bel, 2016. “Decadal climate predictions using sequential
 learning algorithms”. Journal of Climate, 29 (10), 3787–3809, 
 doi:10.1175/JCLI-D-15-0648.1
  """      
  def __init__(self,e=None,w0=None,start=0,nstart=69,delta_eta=10,iters=4,save_w=1):
    self.e=e
    self.w0=w0
    self.start=0
    self.nstart=nstart
    self.delta_eta=delta_eta
    self.iters=iters
    self.save_w=save_w

  def list_params(self):
    print("e")
    print("w0")
    print("start")
    print("nstart")
    print("delta_eta")
    print("iters")

  def print_params(self):
    print("e = "+str(self.e))
    print("w0 = "+str(self.w0))
    print("start = "+str(self.start))
    print("nstart = "+str(self.nstart))
    print("delta_eta = "+str(self.delta_eta))
    print("iters = "+str(self.iters))

  def loss_ega(self,X,y,p):
    l=2*(p-y)*X
    return(l)

  def exp_grad_weight(self,X,y):
    """The weight of the Exponential Gradient Average (EGA) Forecaster without optimization

    Parameters
    ----------
    X : array_like
        The ensemble forecast. For N models and n time steps and ltln locations X size is f(n,N,ltln).
    y : array_like
        The observations y(n,ltln). 
  Returns
  -------
    out: weights
        array of the size (n+1,N,ltln). The last time step is the weight for the future.
    """
    
    e=self.e
    w0=self.w0
    dim=X.shape
    lrt=dim[0]
    N=dim[1]
    dout1 = (dim[0]+1,) + dim[1:]
    pdim = (dim[0]+1,) + dim[2:]
    
    #************************
    # defining weights array
    #************************
    w=np.zeros(X.shape, dtype='Float64')

    #************************
    #bound for the loss 
    #************************
    Y=y[:, np.newaxis,...]
    #M=np.amax(np.abs(self.loss_ega(X,Y,X)),axis=(0,1))
    M=2*np.amax(np.abs(X-Y),axis=(0,1))*np.amax(np.abs(X),axis=(0,1))
    if not hasattr(self, 'M'):
      self.M=M

    #************************************
    #eta and starting weight definitions  
    #************************************
    eta=e
    if (w0 is None) and not hasattr(self, 'w'):
      w_start=np.ones(X.shape[1:], dtype='Float64')/N
    elif hasattr(self, 'w'):
      w_start=self.w[-1].reshape(dim[1:]).astype('float64')
    else:
      w_start=w0.reshape(dim[1:]).astype('float64')
    
    #************************************
    #first time step weight 
    #************************************
    p = np.ones(pdim, dtype='Float64')
    p[0,...]=np.sum(w_start*X[0,...],axis=0)
    if lrt==1:
      los=self.loss_ega(X,Y,p[0,...])/M
      w[0,:]=w_start*np.exp(-eta*los)/np.sum(w_start*np.exp(-eta*los),axis=1)
    else:
      los=self.loss_ega(X[0,:],Y[0,:],p[0,:])/M
      w[0,:]=w_start*np.exp(-eta*los)/np.sum(w_start*np.exp(-eta*los),axis=0)
    
    # equation 3 in strobach and bel (2016)

    #************************************
    #Updating the weights 
    #************************************
    for t in range(1,lrt):
      p[t,:]=np.sum(w[t-1,:]*X[t,:],axis=0)
      los=(self.loss_ega(X[t,:],Y[t,:],p[t,:]))/M
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


  def ega(self,X,y):
    """The weight of the Exponential Gradient Average (EGA) Forecaster with optimization for the learning rate.
    Parameters
    ----------
    X : array_like
        The ensemble forecast. For N models and n time steps and ltln locations f size is f(n,N,ltln...).
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

    dim=X.shape
    lrt=dim[0]
    N=dim[1]
    ltln=int(np.prod(np.array(dim))/lrt/N)
    f1=X.reshape((lrt,N,ltln))
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
    start=self.start # at the beginning eta=0 everywhere
    n=self.nstart #69 # we increase eta n times
    deta=self.delta_eta #10. # each time by deta
    iters=self.iters # 4

    for res in range(0,iters): # 4 iterations
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
        self.e=eta0[:,lp]
        #*************************************************************
        # Running the algorithm for a given eta
        #*************************************************************
        temp_weight=self.exp_grad_weight(f1,y1)

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

    dim1=(dim[0]+1,) + dim[1:]
    weight=weight1.reshape(dim1)
    w0=weight[-1,:]
    self.w0=w0
    if len(dim)>2:
      self.e=eta1.reshape(dim[2:])
    else:
      self.e=eta1

    return(weight)

  def fit(self,X,y):
    """The weight of the Exponential Gradient Average (EGA) Forecaster with optimization for the learning rate.
    This is the outer managment. If the learning rate is given at the initializtion then no optimization for
    eta is performed. In case eta=None (or not provided) the algorithm will search for optimized eta.
    Parameters
    ----------
    X : array_like
        The ensemble forecast. For N models and n time steps and ltln locations f size is f(n,N,ltln).
    y : array_like
        The observations y(n,ltln).        
  Returns
  -------
    out: weights
        array of the size (N,ltln). The last time is the weight for the future.
    """
    e=self.e
    if e is None:
      w=self.ega(X,y)
    else:
      w=self.exp_grad_weight(X,y)
    if self.save_w==1:
      self.w=w.astype('float64')
    else:
      self.w=w[-1].astype('float64')
    return w[-1,:]
    
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
    assert len(X) == len(self.w0)
    assert np.all(self.e)!=None
    X=X[np.newaxis,...]
    y=y[np.newaxis,...]
    w=self.exp_grad_weight(X,y)[1,...]
    if not hasattr(self,"w"):
      self.w=w
    else:
      self.w=np.append(self.w,w[np.newaxis,...],axis=0)
    return w
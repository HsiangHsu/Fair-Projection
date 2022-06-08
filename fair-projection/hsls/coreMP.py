import numpy as np
import scipy as sp
import cvxpy as cp
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)

from tqdm import tqdm
from itertools import islice
import multiprocessing
from multiprocessing import Pool

'''
Implements ADMM solver for model projection
'''

# core ADMM optimization in numpy
def admm(G,y,rho=2,div = 'kl', tol = 1e-6, max_iter = 1000,report=False):
    '''
    Core Model Projection algorithm. Here:
    n - number of data points
    k - number of constraints
    c - number of classes
    
    
    Arguments:
        - G (n x c x k np.array): constraint matrices for each n points
        - y (n x c x 1 np.array): original classifier outputs
        - rho: ADMM parameter
        - div: f-divergence, can be 'kl' or 'cross-entropy'
        - tol: primal-dual gap
        - report: print out primal-dual and feasibility gap
        
    Returns:
        - l (k x 1 np.array): optimal dual parameter lambda
    '''
    
    n,c,k = G.shape
    obj_v = []
    
    # Initialize variables and constants
    
    logy = np.log(y) # used in Step 1
    x = np.zeros((n,c,1)) # initialize x, used in Step 1 
    
    l = np.ones((k,1)) # initialize lambda
    
    v = np.ones((n,c,1)) # initialize v
    
    mu = np.ones((n,c,1)) # initialize mu

    G_t = np.transpose(G, axes=[0, 2, 1])
    # sum of G_i^T \times G_i across batch
    Q = np.sum(G_t @ G, axis = 0)/n
    
    rho2 = rho/2
    
    # optimization variables for step 2
    l_cp = cp.Variable(shape=(k,1),nonneg=True) # lambda for cvx
    d_cp = cp.Parameter(shape=(k,1))
    
    cost = rho2*cp.quad_form(l_cp,Q)+ d_cp.T @l_cp  # cost function for step 2
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective) # quadratic optimization
    
    
    for ix in range(max_iter):
        
        cv = mu + rho*(G @ l) 
    
        ### Step 1: v update ##
        inner_tol = 1e-13
        
        # kl-divergence
        if div == 'kl':
            a = cv - rho*logy
            x = np.zeros((n,c,1))

            for jx in range(50): # TODO: check this limit
                xold = x
                x= -(sp.special.softmax(x,axis=1)+a)/rho 
                
                # check if update is small
                if np.abs(x-xold).max()<inner_tol:
                    break

            v = (x-logy) # update v

        
        # cross-entropy
        elif div == 'cross-entropy':
            # initialize z
            z = np.zeros((n,1,1))
            a = 4*rho*y
            
            for jx in range(50): # TODO: check this limit
                cpz = cv + z
                b = np.sqrt(a + cpz*cpz)
                num = (-cpz + b)
                gz = (-1 + num.sum(axis=1)*0.5).reshape(n,1,1)
                gprime = -0.5*(num/b).sum(axis=1).reshape(n,1,1)
                z = z - gz/gprime
                
                if np.abs(gz/gprime).max() < inner_tol: #break if update is small
                    break

            cpz = cv + z
            x = .5*(-cpz + np.sqrt(a+cpz*cpz))
            
            v = -(x+cv)/rho # update v
    
            
        # TODO: raise error if divergence not listed
            
        ### Step 2: lambda update ###
        d_cp.value = tf.reduce_sum( G_t @ (mu + rho*v),axis=0).numpy()/n # linear part
        prob.solve(warm_start=True)
        
        l_old = l
             
        l = l_cp.value # assign value to tensorflow variable

        ### Step 3: mu update ###
        mu +=  rho*(v + (G @ l) )
        
    # report gaps
    
    if report:

        # compute primal classifier
        h = predict(l,G,y,div = div)
        infeas = ((np.transpose(G,axes=[0,2,1])@np.array(h)).sum(axis=0)/n).max()
        # print('Max infeasibility: '+str(infeas))
        
        if div == 'kl':
            obj = -np.sum(sp.special.logsumexp(x,1))/n
            error = 100*np.abs((sp.special.kl_div(h,y).sum()/n-obj)/obj)
            
        elif div == 'cross-entropy':
            _, obj = predict_cross(-v,y,return_obj=True)
            obj = obj.sum()/n
            error = 100*np.abs((sp.special.kl_div(y,h).sum()/n-obj)/obj)

        # print('Error (percentage of dual): ' + str(error))

    return l


# core ADMM optimization in TF
def admm_tf(G,y,rho=2,div = 'kl', max_iter = 1000, eps1=1e-7, eps2=1e-2, eps3=1e-7):
    '''
    Core Model Projection algorithm. Here:
    n - number of data points
    k - number of constraints
    c - number of classes
    
    
    Arguments:
        - G (n x c x k np.array): constraint matrices for each n points
        - y (n x c x 1 np.array): original classifier outputs
        - rho: ADMM parameter
        - div: f-divergence, can be 'kl' or 'cross-entropy'
        - tol: primal-dual gap
        - report: print out primal-dual and feasibility gap
        
    Returns:
        - l (k x 1 np.array): optimal dual parameter lambda
    '''
    
    n,c,k = G.shape
    reg = 1 / np.sqrt(n)

    # Create tensforflow variables and constants
    G_tf = tf.constant(G)
    # G_tf_t = tf.transpose(G_tf, perm=[0, 2, 1])
    
    logy_tf = tf.constant(np.log(y)) # used in Step 1
    y_tf = tf.constant(y) # used in Step 1
    
    x_tf = tf.Variable(np.zeros((n,c,1)),trainable = False) # initialize x, used in Step 1 
    x_tf_old = tf.Variable(np.zeros((n,c,1)),trainable = False) # initialize x, used in Step 1 
    
    l_tf = tf.Variable(np.ones((k,1)), trainable=False) # initialize lambda
    
    v_tf = tf.Variable(np.ones((n,c,1)), trainable=False) # initialize v
    
    mu_tf = tf.Variable(np.ones((n,c,1)), trainable=False) # initialize mu
    
    z_tf = tf.Variable(np.zeros((n,1,1)), trainable=False) # used in step 1 from  cross-entropy
    
    c_tf = tf.Variable(np.ones((n,c,1)), trainable=False) # initialize c
    
    Gl_tf = tf.Variable(np.ones((n,c,1)), trainable=False) # initialize variable of G times lambda (needed for memory)
    
    
    # sum of G_i^T \times G_i across batch
    # Q = tf.reduce_sum(G_tf_t @ G_tf, axis = 0).numpy()/n # this way of computing Q will lead to OOM error for large batches
    
    n_list = range(n)
    it = iter(n_list)
    BATCH_SIZE = 100000 # batch size for reducing G product, this may have to be changed depending on memory
    ln = list(iter(lambda: tuple(islice(it, BATCH_SIZE)), ())) #TODO: there must be a smarter way of computing these indices...
#     Qb = tf.Variable(np.zeros((k,k)))
    
#     for ix in ln:
#         Qb.assign_add(tf.reduce_sum(tf.linalg.matmul(G_tf[ix[0]:ix[-1]+1,:,:],G_tf[ix[0]:ix[-1]+1,:,:],transpose_a=True), axis = 0))

    Qb = tf.Variable(np.zeros((k,k)))
    
    
    
    Gdataset = tf.data.Dataset.from_tensor_slices(G).batch(BATCH_SIZE)
    
    for Gb in Gdataset:
        Qb.assign_add(tf.reduce_sum(tf.linalg.matmul(Gb,Gb,transpose_a=True), axis = 0))

    rho2 = rho / 2
    Q = (rho2*Qb/n).numpy() + reg/2*np.eye(k)

    # optimization variables for step 2
    l_cp = cp.Variable(shape=(k,1),nonneg=True) # lambda for cvx
    d_cp = cp.Parameter(shape=(k,1))
    
    cost = cp.quad_form(l_cp,Q)+ d_cp.T @l_cp  # cost function for step 2
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective) # quadratic optimization
    
    # pre-compute G times l
    Gl_tf.assign(G_tf @ l_tf)
#     for ix in ln:
#         Gl_tf_slice = Gl_tf[ix[0]:ix[-1]+1]
#         Gl_tf_slice.assign(tf.linalg.matmul(G_tf[ix[0]:ix[-1]+1,:,:],l_tf))


    # print(n, c, k)
    # G_stack = np.vstack([G[i, :, :] for i in range(n)])
    # print('G Rank: ', np.linalg.matrix_rank(G_stack))
    #
    # Z = sp.linalg.null_space(G_stack)
    # GZ = np.matmul(G_stack, Z)
    #
    # print('Z', Z)
    # for i in range(5):
    #     print('GZ', GZ[i, :])


    inner_tol = 1e-10
    for ix in range(max_iter):
        ## working case: (v l m)*iter
        ## now: v (l v m)*iter
        ### Step 1: v update ###
        c_tf.assign(mu_tf + rho * Gl_tf)
        # kl-divergence
        if div == 'kl':
            ## l2 regularizaton => rho = rho + a small number to be determined, e.g., 1/sqrt(sample size)
            a_tf = c_tf - rho*logy_tf
            x_tf.assign(np.zeros((n,c,1)))

            for jx in range(100): # TODO: check this limit
                x_tf_old.assign(x_tf)
                
                x_tf.assign(-(tf.nn.softmax(x_tf,axis=1)+a_tf)/(rho + reg))
                
                # check if update is small
                if tf.math.reduce_max(tf.math.abs(x_tf-x_tf_old)).numpy()<inner_tol:
                    break

            v_tf.assign(x_tf-logy_tf) # update v_tf
        
        # cross-entropy
        elif div == 'cross-entropy':
            # initialize z
            z_tf.assign(np.zeros((n,1,1)))
            a_tf = 4*(rho + reg)*y_tf
            
            for jx in range(100): # TODO: check this limit
                cpz = c_tf + z_tf
                b = tf.math.sqrt(a_tf + cpz*cpz)
                num = (-cpz + b)
                gz = (-1 + tf.math.reduce_sum(num,1,keepdims=True)*0.5)
                gprime = -0.5*tf.math.reduce_sum(num/b,1,keepdims=True)
                inc = gz/gprime
                z_tf.assign_sub(inc)
                
                if tf.math.reduce_max(tf.math.abs(inc)) < inner_tol: #break if update is small
                    break

            cpz = c_tf + z_tf
            x_tf.assign(.5*(-cpz + tf.math.sqrt(a_tf+cpz*cpz)))
            
            v_tf.assign(-(x_tf+c_tf)/(rho + reg)) # update v

        # ### Step 2: lambda update ###
        d_cp.value = tf.reduce_sum( tf.linalg.matmul(G_tf, (mu_tf + rho*v_tf),transpose_a=True ),axis=0).numpy()/n # linear part ## rho -> rho+reg?
        prob.solve(warm_start=True)
        l_tf.assign(l_cp.value) # assign value to tensorflow variable

        ### Step 3: mu update ###
        Gl_tf.assign(G_tf @ l_tf)
        mu_tf.assign_add((rho + reg)*(v_tf + Gl_tf ))
        
    # if report:
    if ix % 50 == 0:
        # compute primal classifier
        h = predict(l_tf.numpy(),G,y,div = div)
        max_infeasibility = ((np.transpose(G,axes=[0,2,1])@np.array(h)).sum(axis=0)/n).max()

        # print('Max infeasibility: '+str(infeas))

        if div == 'kl':
            obj =  -tf.math.reduce_sum(tf.math.reduce_logsumexp(x_tf,1)).numpy()/n
            error = 100*np.abs((sp.special.kl_div(h,y).sum()/n-obj)/obj)

        elif div == 'cross-entropy':
            _, obj = predict_cross(-v_tf.numpy(),y,return_obj=True)
            obj = obj.sum()/n
            error = 100*np.abs((sp.special.kl_div(y,h).sum()/n-obj)/obj)

        percentage_error = error
        absolute_error = obj*error/100

        if (max_infeasibility < eps1) and ((percentage_error < eps2) or (absolute_error < eps3)):
            return l_tf.numpy()

        # print('Error (percentage of dual): ' + str(error))
        # print('Error (absolute gap): ' + str(obj*error/100))
        
        
    return l_tf.numpy()

# core ADMM optimization in TF
def admm_tf_batch(G,y,rho=2,div = 'kl', tol = 1e-6, max_iter = 1000,report=False,BATCH_SIZE=10000):
    '''
    Core Model Projection algorithm. Here:
    n - number of data points
    k - number of constraints
    c - number of classes
    
    
    Arguments:
        - G (n x c x k np.array): constraint matrices for each n points
        - y (n x c x 1 np.array): original classifier outputs
        - rho: ADMM parameter
        - div: f-divergence, can be 'kl' or 'cross-entropy'
        - tol: primal-dual gap
        - report: print out primal-dual and feasibility gap
        
    Returns:
        - l (k x 1 np.array): optimal dual parameter lambda
    '''
    
    n,c,k = G.shape
    
    # Create tensforflow variables and constants
    
    l_tf = tf.Variable(np.ones((k,1)), trainable=False) # initialize lambda
    
    rho_tf = tf.constant(rho,dtype=tf.float64)


    # Batch indexing stuff
    n_list = range(n)
    it = iter(n_list)
   
    ln = list(iter(lambda: tuple(islice(it, BATCH_SIZE)), ())) #TODO: there must be a smarter way of computing these indices...

    index = [(ix[0],ix[-1]+1) for ix in ln]
    index_tf = [tf.range(ix[0],ix[1]) for ix in index]

    ###### Aux Functions for ADMM #####
    @tf.function
    def mul_self_transpose(G):
        return tf.reduce_sum(tf.matmul(G,G,transpose_a=True),axis=0)

    
    ####### End of aux functions #####
    
    # create list of variables
    Glist = [tf.convert_to_tensor(G[ix[0]:ix[1]]) for ix in index]
        
    # mu
    mulist = [tf.Variable(np.ones((len(ix),c,1)),trainable=False) for ix in ln]
    
    # v
    vlist = [tf.Variable(np.ones((len(ix),c,1)),trainable=False) for ix in ln]
    


    Qb = tf.Variable(np.zeros((k,k)))
    
    
    n_list2 = range(n)
    it2 = iter(n_list2)
   
    ln2 = list(iter(lambda: tuple(islice(it2, 10000)), ())) #TODO: there must be a smarter way of computing these indices...

    index2 = [(ix[0],ix[-1]+1) for ix in ln2]

    
    for ia in index2:
        Qb.assign_add(mul_self_transpose(G[ia[0]:ia[1]]))


    Q = (Qb/n).numpy()

    rho2 = rho/2
    
    # optimization variables for step 2
    l_cp = cp.Variable(shape=(k,1),nonneg=True) # lambda for cvx
    d_cp = cp.Parameter(shape=(k,1))
    
    cost = rho2*cp.quad_form(l_cp,Q)+ d_cp.T @l_cp  # cost function for step 2
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective) # quadratic optimization
    
    if div == 'cross-entropy':
        step1 = step1_cross()
        ylist = [tf.convert_to_tensor(y[ix[0]:ix[1]]) for ix in index]

        
    elif div == 'kl':
        step1 = step1_kl()
        ylist = [tf.convert_to_tensor(np.log(y[ix[0]:ix[1]])) for ix in index]
        
        
    not_first_iter = tf.Variable(False)
    
    for jx in range(max_iter):
        
        
        ## TODO: move all vriables to a list
        d_cp.value = np.zeros((k,1))
        
        ### Step 1: batch v update ###
        for (mu_tf,G_tf,y_tf,v_tf) in zip(mulist,Glist,ylist,vlist):               
            #v_new, d_new, mu_new = step1(mu_tf,G_tf,l_tf,rho_tf,y_tf,v_tf,not_first_iter)
            d_new = step1(mu_tf,G_tf,l_tf,rho_tf,y_tf,v_tf,not_first_iter)

            #v_tf.assign(v_new)
            
            # Step 3 update
            #mu_tf.assign(mu_new)
        
            
            # step 2 update
            d_cp.value += d_new.numpy()/n
            


        not_first_iter.assign(True)    # not first iteration anymore

        ## Solve for step 2
        prob.solve(warm_start=True)
        
        l_old = l_tf.numpy()
             
        l_tf.assign(l_cp.value) # assign value to tensorflow variable


        
        
    if report: 
        # compute v
        v_np = [v_tf.numpy() for v_tf in vlist]
        v_np = np.concatenate(v_np)
            
        
        # compute primal classifier
        h = predict(l_tf.numpy(),G,y,div = div)
        infeas = ((np.transpose(G,axes=[0,2,1])@np.array(h)).sum(axis=0)/n).max()
        
        # print('Max infeasibility: '+str(infeas))
        
        if div == 'kl':
            obj =  -tf.math.reduce_sum(tf.math.reduce_logsumexp(v_np+np.log(y),1)).numpy()/n
            error = 100*np.abs((sp.special.kl_div(h,y).sum()/n-obj)/obj)
            
        elif div == 'cross-entropy':
            _, obj = predict_cross(-v_np,y,return_obj=True)
            obj = obj.sum()/n
            error = 100*np.abs((sp.special.kl_div(y,h).sum()/n-obj)/obj)
            
        # print('Error (percentage of dual): ' + str(error))
        
        
    return l_tf.numpy()


# predict cvx
def predict_cvx(l,G,y,div='kl'):
    '''
    Compute the corrected classifier output using CVX.
    
    
    Arguments:
        - l (k x 1 np.array): dual parameter lambda
        - G (c x k np.array): constraint matrix for the given data point 
        - y (c x 1 np.array): original classifier output
        - div: f-divergence, can be 'kl' or 'cross-entropy'
        
    Retruns:
        - h (c x 1 np.array): corrected prediction
    '''
    
    # create optimization variable
    h = cp.Variable(shape=y.shape,nonneg=True)
        
    # simplex and cost function
    constraints = [sum(h)==1]
    
    if div == 'kl':        
        # kl cost
        cost = sum(cp.kl_div(h,y))+(G.dot(l)).transpose()@h
    
    elif div == 'cross-entropy':
        # cross-entropy cost
        cost = sum(cp.kl_div(y,h))+(G.dot(l)).transpose()@h
        
    
    # solve using cvxpy
    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    return h.value


def predict(l,G,y,div='kl'):
    '''
    Compute the corrected classifier output.
    
    
    Arguments:
        - l (k x 1 np.array): dual parameter lambda
        - G (n x c x k np.array): constraint matrix for the given data point 
        - y (n x c x 1 np.array): original classifier output
        - div: f-divergence, can be 'kl' or 'cross-entropy'
        
    Retruns:
        - h (n x c x 1 np.array): corrected prediction
    '''
    
    # create optimization variable
    n,c,k = G.shape
        
    # compute v
    v = G @ l
     
    
    if div == 'kl':        
        # kl cost
        h = sp.special.softmax(-v + np.log(y),axis=1)
    
    elif div == 'cross-entropy':
        # cross-entropy cost
        
        if n< 5000:
            h = predict_cross(v,y)         
        # if batch size is large, use multiprocess
        else:
            cores = multiprocessing.cpu_count() -1
            # create batches of size 100
            n_list = range(n)
            it = iter(n_list)
            size = 100
            ln = list(iter(lambda: tuple(islice(it, size)), ())) # list of indices
            
            vals = [(v[ix,:,:],y[ix,:,:]) for ix in ln] # split of v and y values
            
            #compute in parallel
            with Pool(cores) as p:
                hvals = (p.starmap(predict_cross, [(v[ix,:,:],y[ix,:,:]) for ix in ln]))
                
            h = np.concatenate(hvals,axis=0)
    
    return h


def predict_cross(v,y,tol=1e-10,alpha=.3,beta=.5,max_iter = 100,return_obj = False):
    '''
    Interior-point method for computing corrected classifier with cross-entropy objective.
    This is essentially algorithm 10.1 in Boyd and Vandenberghe.
    As usual, n is the batch size.
    
    Arguments:
    - v (n x c x 1 np.array): linear term in the conjugate
    - y (n x c x 1 np.array): original classifier output
    - tol: worst-case batch relative error between objective and optimal
    - alpha, beta: line-search parameters (see CVX book, Algorithm 9.2)
    - max_iter: maximum number of iterations
    - return_obj: if objective should be returned as well as a second argument
    
    Returns:
    - h (n x c x 1 np.array): corrected predictions
    
    '''
    
    yinv = 1/(y+tol) # we will frequently use y inverse, so we pre-compute to avoid division by 0
    n,c,_ = y.shape
    
    ############### auxiliary functions ##############
    def newtonStep(h):
        a = h*yinv
        b = h*a
        fp = v - (1/a)
        w = -np.sum(fp*b,axis=1)/np.sum(b,axis=1)
        w = w.reshape(n,1,1)
        step = -(fp+w)*b
        return step

    # objective
    def f(h):
        cr = np.sum(v*h,axis=1) -np.sum(y*np.log(h),axis=1)
        return cr

    # grad
    def fp(h):
        return v - y/h

    # compute newton decrement (10.12 in Boyd's book)
    def newton_decrement(h,step):
        lx = np.sqrt(np.sum(step*step*y/(h*h),axis=1))
        return lx.max()

    # vectorized line search
    def line_search(h,step,alpha=.25,beta=.5):
        t = np.ones((n,1,1)) # initialize t

        # make sure no entry becomes negative
        while True:
            hnew = h+step*t
            ix = (hnew.min(axis=1)<0)
            if sum(ix) == 0:
                break
            else:
                t[ix] = t[ix]*beta

        # now search until break condition is met
        delta = (fp(h)*step).sum(axis=1).reshape(n,1,1)
        obj = f(h).reshape(n,1,1)


        while True:
            hnew = h + step*t
            obj_inc = f(hnew).reshape(n,1,1)
            ix = (obj_inc > obj + alpha*t*delta)
            if sum(ix) == 0:
                break
            else:
                t[ix] = t[ix]*beta          

        return hnew
    ############### end of auxiliary functions ##############
    
    # main procedure
    # initialize at y
    h = y
    obj = f(h)
    
    # Newton's method
    for j in range(max_iter):
        step = newtonStep(h)
        min_dec = newton_decrement(h,step)

        if min_dec**2/2<tol:
            break
        else:
            h = line_search(h,step,alpha=alpha,beta=beta)

            obj = f(h) + np.sum(y*np.log(y),axis=1)
        
    if return_obj:
        return h, obj
    else:
        return h
    
### TF implementations
# step 1 for kl
class step1_kl(tf.Module):
    
    def __init__(self):
        self.x = None
        self.x_old = None
        
        
    @tf.function
    def __call__(self,mu,G,l,rho,logy,v,not_first_iter,inner_tol=tf.constant(1e-10,dtype=tf.float64)):
        
        if self.x is None:
            self.x = tf.Variable(tf.zeros_like(logy,dtype=tf.float64),shape=tf.TensorShape([None,logy.shape[1],1 ]))
            self.x_old = tf.Variable(tf.ones_like(logy,dtype=tf.float64),shape=tf.TensorShape([None,logy.shape[1],1 ]))
            self.k = tf.Variable(0,dtype=tf.float64)
            self.prec = tf.Variable(1,dtype=tf.float64)
            
        else:
            self.x.assign(tf.zeros_like(logy,dtype=tf.float64))
            self.x_old.assign(tf.ones_like(logy,dtype=tf.float64))
            self.k.assign(0)
            self.prec.assign(1) 
            #
            
        Gl = tf.matmul(G,l)
        if not_first_iter:
            mu.assign_add(rho*(v+ Gl))


        a = mu+rho*Gl - rho*logy
        
        while tf.math.logical_or(tf.greater(self.prec, inner_tol),tf.less(self.k,tf.constant(100,dtype=tf.float64))):

            self.x_old.assign(self.x)

            self.x.assign(-(tf.nn.softmax(self.x,axis=1)+a)/rho )
            self.k.assign_add(1)
            self.prec.assign(tf.math.reduce_max(tf.math.abs(self.x-self.x_old)))


        v.assign(self.x-logy)
        
        d = tf.reduce_sum( tf.linalg.matmul(G, (mu + rho*v),transpose_a=True ),axis=0)

        return d

# step 1 for cross-entropy
class step1_cross(tf.Module):
  def __init__(self):
    self.z = None
    self.prec=None


  @tf.function
  def __call__(self,mu,G,l,rho,y,v,not_first_iter,inner_tol=tf.constant(1e-10,dtype=tf.float64)):
    if self.z is None:
        self.z = tf.Variable(tf.zeros([y.shape[0],1,1],dtype=tf.float64),shape= tf.TensorShape(None))
        self.prec = tf.Variable(1,dtype=tf.float64)
        self.k = tf.Variable(0,dtype=tf.float64)
    else:
        self.z.assign(tf.zeros([y.shape[0],1,1],dtype=tf.float64))
        self.prec.assign(1)
        self.k.assign(0)
        #
        
    Gl = tf.matmul(G,l)
    
    # compute step 3 in iter
    if not_first_iter:
        mu.assign_add(rho*(v+ Gl))

       
    
    c = mu+rho*Gl
    
    a = 4*rho*y

    while tf.math.logical_or(tf.greater(self.prec, inner_tol),tf.less(self.k,tf.constant(100,dtype=tf.float64))):
        cpz = c + self.z
        b = tf.math.sqrt(a + cpz*cpz)
        num = (-cpz + b)
        gz = (-1 + tf.math.reduce_sum(num,1,keepdims=True)*0.5)
        gprime = -0.5*tf.math.reduce_sum(num/b,1,keepdims=True)
        inc = (gz/gprime)
        self.z.assign_sub(inc)
        self.prec.assign(tf.math.reduce_max(tf.math.abs(inc)))
        self.k.assign_add(1)

    cpz = c + self.z
    x = (.5*(-cpz + tf.math.sqrt(a+cpz*cpz)))
    v.assign(-(x+c)/rho) 
    
    d = tf.reduce_sum( tf.linalg.matmul(G, (mu + rho*v),transpose_a=True ),axis=0)

    return d
 
    
#     # core ADMM optimization in TF
# def admm_tf_batch2(G,y,rho=2,div = 'kl', tol = 1e-6, max_iter = 1000,report=False,BATCH_SIZE=10000):
#     '''
#     Core Model Projection algorithm. Here:
#     n - number of data points
#     k - number of constraints
#     c - number of classes
    
    
#     Arguments:
#         - G (n x c x k np.array): constraint matrices for each n points
#         - y (n x c x 1 np.array): original classifier outputs
#         - rho: ADMM parameter
#         - div: f-divergence, can be 'kl' or 'cross-entropy'
#         - tol: primal-dual gap
#         - report: print out primal-dual and feasibility gap
        
#     Returns:
#         - l (k x 1 np.array): optimal dual parameter lambda
#     '''
    
#     n,c,k = G.shape
    
#     # Create tensforflow variables and constants
    
#     l_tf = tf.Variable(np.ones((k,1)), trainable=False) # initialize lambda
    
#     rho_tf = tf.constant(rho,dtype=tf.float64)


#     # Batch indexing stuff
#     n_list = range(n)
#     it = iter(n_list)
   
#     ln = list(iter(lambda: tuple(islice(it, BATCH_SIZE)), ())) #TODO: there must be a smarter way of computing these indices...

#     index = [(ix[0],ix[-1]+1) for ix in ln]
#     index_tf = [tf.range(ix[0],ix[1]) for ix in index]

#     ###### Aux Functions for ADMM #####
#     @tf.function
#     def mul_self_transpose(G):
#         return tf.reduce_sum(tf.matmul(G,G,transpose_a=True),axis=0)

    
#     ####### End of aux functions #####
    
#     # create list of variables
#     Glist = [tf.convert_to_tensor(G[ix[0]:ix[1]]) for ix in index]
        
#     # mu
#     mulist = [tf.Variable(np.ones((len(ix),c,1)),trainable=False) for ix in ln]
    
#     # v
#     vlist = [tf.Variable(np.ones((len(ix),c,1)),trainable=False) for ix in ln]
    
#     # x
#     xlist = [tf.Variable(np.ones((len(ix),c,1)),trainable=False) for ix in ln]# initialize x, used in Step 1 
    
    
#     # x old
#     x_oldlist = [tf.Variable(np.ones((len(ix),c,1)),trainable=False) for ix in ln] # initialize x, used in Step 1 
    
#     # z
#     zlist = [tf.Variable(np.ones((len(ix),1,1)),trainable=False) for ix in ln]# initialize x, used in Step 1 


#     Qb = tf.Variable(np.zeros((k,k)))
    
    
#     n_list2 = range(n)
#     it2 = iter(n_list2)
   
#     ln2 = list(iter(lambda: tuple(islice(it2, 10000)), ())) #TODO: there must be a smarter way of computing these indices...

#     index2 = [(ix[0],ix[-1]+1) for ix in ln2]

    
#     for ia in index2:
#         Qb.assign_add(mul_self_transpose(G[ia[0]:ia[1]]))


#     Q = (Qb/n).numpy()

#     rho2 = rho/2
    
#     # optimization variables for step 2
#     l_cp = cp.Variable(shape=(k,1),nonneg=True) # lambda for cvx
#     d_cp = cp.Parameter(shape=(k,1))
    
#     cost = rho2*cp.quad_form(l_cp,Q)+ d_cp.T @l_cp  # cost function for step 2
#     objective = cp.Minimize(cost)
#     prob = cp.Problem(objective,cost) # quadratic optimization
    
#     if div == 'cross-entropy':
#         step1 = step1_cross()
#         ylist = [tf.convert_to_tensor(y[ix[0]:ix[1]]) for ix in index]

        
#     elif div == 'kl':
#         step1 = step1_kl()
#         ylist = [tf.convert_to_tensor(np.log(y[ix[0]:ix[1]])) for ix in index]
        
        
#     not_first_iter = tf.Variable(False)
    
#     inner_tol = 1e-10 # stopping criteria for inner iteration
    
#     for jx in tqdm(range(max_iter)):
        
#         d_cp.value = np.zeros((k,1))
        
#         for (mu_tf,G_tf,y_tf,v_tf,x_tf,x_tf_old,z_tf) in zip(mulist,Glist,ylist,vlist,xlist,x_oldlist,zlist):    
            
#             Gl_tf = tf.matmul(G_tf,l_tf)
            
            
#             ### Step 3: v update ###
#             if jx != 0:
#                 mu_tf.assign_add(rho*(v_tf+ Gl_tf))
            
#             ### Step 1: mu update
#             c_tf = mu_tf + rho*Gl_tf 
#             # kl-divergence
#             if div == 'kl':
#                 a_tf = c_tf - rho*y_tf
#                 x_tf.assign(tf.zeros_like(x_tf))

#                 for jx in range(100): # TODO: check this limit
#                     x_tf_old.assign(x_tf)

#                     x_tf.assign(-(tf.nn.softmax(x_tf,axis=1)+a_tf)/rho )

#                     # check if update is small
#                     if tf.math.reduce_max(tf.math.abs(x_tf-x_tf_old)).numpy()<inner_tol:
#                         break

#                 v_tf.assign(x_tf-y_tf) # update v_tf
        
#             # cross-entropy
#             elif div == 'cross-entropy':
#                 # initialize z
#                 z_tf.assign(tf.zeros_like(z_tf))
#                 a_tf = 4*rho*y_tf

#                 for jx in range(100): # TODO: check this limit
#                     cpz = c_tf + z_tf
#                     b = tf.math.sqrt(a_tf + cpz*cpz)
#                     num = (-cpz + b)
#                     gz = (-1 + tf.math.reduce_sum(num,1,keepdims=True)*0.5)
#                     gprime = -0.5*tf.math.reduce_sum(num/b,1,keepdims=True)
#                     inc = gz/gprime
#                     z_tf.assign_sub(inc)

#                     if tf.math.reduce_max(tf.math.abs(inc)) < inner_tol: #break if update is small
#                         break

#                 cpz = c_tf + z_tf
#                 x_tf.assign(.5*(-cpz + tf.math.sqrt(a_tf+cpz*cpz)))

#                 v_tf.assign(-(x_tf+c_tf)/rho) # update v
            
#             # update value
#             d_cp.value += tf.reduce_sum( tf.linalg.matmul(G_tf, (mu_tf + rho*v_tf),transpose_a=True ),axis=0).numpy()/n # linear part

#         ### Step 2: lambda update ###
        
#         prob.solve(warm_start=True)
        
#         l_old = l_tf.numpy()
             
#         l_tf.assign(l_cp.value) # assign value to tensorflow variable
        
        
#     if report: 
#         # compute v
#         v_np = [v_tf.numpy() for v_tf in vlist]
#         v_np = np.concatenate(v_np)
            
        
#         # compute primal classifier
#         h = predict(l_tf.numpy(),G,y,div = div)
#         infeas = ((np.transpose(G,axes=[0,2,1])@np.array(h)).sum(axis=0)/n).max()
        
#         print('Max infeasibility: '+str(infeas))
        
#         if div == 'kl':
#             obj =  -tf.math.reduce_sum(tf.math.reduce_logsumexp(v_np+np.log(y),1)).numpy()/n
#             error = 100*np.abs((sp.special.kl_div(h,y).sum()/n-obj)/obj)
            
#         elif div == 'cross-entropy':
#             _, obj = predict_cross(-v_np,y,return_obj=True)
#             obj = obj.sum()/n
#             error = 100*np.abs((sp.special.kl_div(y,h).sum()/n-obj)/obj)
            
#         print('Error (percentage of dual): ' + str(error))     
        
        
#     return l_tf.numpy()

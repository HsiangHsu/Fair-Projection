import numpy as np
import sys
sys.path.insert(1, '../code/')
import coreMP as MP
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


'''
  Group fairness class.
'''

class GFair:
    
    def __init__(self,clf_Y,clf_S=None,clf_SgY=None,div='kl'):
        '''
          Class initializer.
          Args:
          
          * clf_Y = base model that will be used to predict outcome Y
          * clf_S = model that will be used to predict sensitive attributes
          * clf_SgY= model that will be used to predict sensitive attribute from X and Y
          

          
          Both models above must have fit/predict method.
        '''
        self.clf_S = clf_S
        self.clf_SgY = clf_SgY
        self.clf_Y = clf_Y
        self.Trained = False
        self.Projected = False
        self.div = div
        
    
    def fit(self, X,y,s, sample_weight):
        '''
        Fit models for model projection.
        Three models will be fit:
        * Py_x = predicts Y from X and is the model that will be projected
        * Ps_x = predicts S from X. Used for SP. Only trained if not None in class initialization.
                 Returns one-hot encoded matrix if not given for S if model is None.
        * Ps_xy = predicts S from Y and X. Only trained if not None in class initialization.
                Returns one-hot encoded matrix if not given for S if model is None
        
        
        Args (same format received by sklearn model)
        
        * X = feature array
        * y = output array
        * s = group attribute array
        '''
        # print('...Training base model...')
        
        
        # compute estimate of marginals
        self.Pys = pd.crosstab(y,s,rownames='Y',colnames='S')/len(y)
        
        
        # create list of categorical features
        self.y_categories_ = np.array(list(self.Pys.index))
        #self.ys_categories_ = []
        self.s_categories_ = np.array(list(self.Pys))
        
                
        # one-hot-encode y
        self.enc_y = OneHotEncoder(handle_unknown='ignore',categories = [self.y_categories_],sparse=False)
        yo = self.enc_y.fit_transform(y.reshape(-1,1))
        
        # fit for y
        self.clf_Y.fit(X,np.argmax(yo,axis=1), sample_weight=sample_weight)
        
        # one-hot-encode s
        self.enc_s = OneHotEncoder(handle_unknown='ignore',categories = [self.s_categories_],sparse=False)
        so = self.enc_s.fit_transform(s.reshape(-1,1))

        
        if not (self.clf_S is None):
            # print('...Training model for predicting S from X...')
            self.clf_S.fit(X,np.argmax(so,axis=1), sample_weight=sample_weight)
            
        if not (self.clf_SgY is None):
            # print('...Training model for predicting S from X and Y...')
            self.clf_SgY.fit(np.concatenate((X,y.reshape(-1,1)),axis=1),np.argmax(so,axis=1), sample_weight=sample_weight)
            
        
        
        self.Trained=True
        
        
    
    def buildG(self, X,constraints,y=None,s=None):
        '''
        Build constraint matrix. We will need to perturb Py_x so it is in the middle of the simplex.
        '''
        fudge = 1e-4

        
        assert self.Trained,  "Fit models first!"
        
        if y is None:
            # if  y is not given, use trained models
            Py_x = self.clf_Y.predict_proba(X)
            self.Py_x = Py_x

            
        else:
            #ys = np.array(['-'.join([str(yx),str(sx)]) for (yx,sx) in zip(y,s) ])
            #self.Pys_x = self.enc_ys.fit_transform(ys.reshape(-1,1))
            
            # use one-hot encoding of y to build matrix
            Py_x = self.enc_y.fit_transform(y.reshape(-1,1))
            
        
        # reshape Pys_x
        #self.Pys_x = self.Pys_x.reshape(len(self.Pys_x),self.Pys.shape[0],self.Pys.shape[1])
            
        # add fudge
        Py_x = (Py_x+fudge)/((Py_x+fudge).sum(axis=1,keepdims=True)) ## constant fudge
        ################################################
        # Py_x = Py_x + 1e-9 * np.random.uniform(low=1e-1, high=1.0, size=Py_x.shape) ##
        # Py_x = Py_x / Py_x.sum(axis=1, keepdims=True) ##
        ################################################

        # compute marginals
        Py = self.Pys.sum(axis=1).to_numpy()
  
                
        # normalize marginal
        #normPy_x = Py_x/Py.reshape(1,len(Py))
                
        # useful constants
        y_len = len(self.y_categories_)    
        s_len = len(self.s_categories_)    
        n_samples = X.shape[0]


        
        Glist = [] # list for storing constraints
        
        for (constraint,alpha) in constraints:
            if constraint == 'meo':

                
                # if s not given, use trained model
                if s is None:
                    assert not (self.clf_SgY is None), "Fit classifier for predicting S from X and Y first!"
                

                for (yv,y_ix) in zip(self.y_categories_,range(y_len)):
                    
                    # initialize constraint matrix
                    Gp = np.zeros((n_samples, y_len,s_len))
                    Gm = np.zeros((n_samples, y_len,s_len))
                    
                    # prepare proabilities of group membership
                    if s is None:
                        y = np.array([yv for i in range(n_samples)])
                        Xy = np.concatenate((X,y.reshape(-1,1)),axis=1)
                        ## X: (n, m), y: (n, c), Xy: (n, m+c) Ps_xy: (n, d)
                        
                        Ps_xy = self.clf_SgY.predict_proba(Xy)

                    else:
                        Ps_xy = self.enc_s.fit_transform(s.reshape(-1,1))

                    ################################################
                    # Ps_xy = Ps_xy + 1e-9 * np.random.uniform(low=1e-1, high=1.0, size=Ps_xy.shape)  ## fudge
                    # Ps_xy = Ps_xy / Ps_xy.sum(axis=1, keepdims=True)  ## fudge
                    ################################################

                    for (sv,s_ix) in zip(self.s_categories_,range(s_len)):
                        
                        # upper constraint
                        Gp[:,y_ix,s_ix] = Py_x[:,y_ix]*( (Ps_xy[:,s_ix]/self.Pys.loc[yv,sv])-((1+alpha)/Py[y_ix]  ) )
                        
                        # lower constraint
                        Gm[:,y_ix,s_ix] = Py_x[:,y_ix]*( -(Ps_xy[:,s_ix]/self.Pys.loc[yv,sv])+((1-alpha)/Py[y_ix]  ) )
                        
                    Glist.append(Gp)
                    Glist.append(Gm)
                    
            if constraint == 'sp':

                # if s not given, use trained model
                if s is None:
                    assert not (self.clf_S is None), "Fit classifier for predicting S from X and Y first!"
                

                for (yv,y_ix) in zip(self.y_categories_,range(y_len)):
                    
                    # initialize constraint matrix
                    Gp = np.zeros((n_samples, y_len,s_len))
                    Gm = np.zeros((n_samples, y_len,s_len))
                    
                    # prepare proabilities of group membership
                    if s is None:
                        
                        Ps_x = self.clf_S.predict_proba(X)
                        
                    else:
                        Ps_x = self.enc_s.fit_transform(s.reshape(-1,1))

                    ################################################
                    # Ps_x = Ps_x + 1e-9 * np.random.uniform(low=1e-1, high=1.0, size=Ps_x.shape) ## fudge
                    # Ps_x = Ps_x / Ps_x.sum(axis=1, keepdims=True)  ## fudge
                    ################################################

                    for (sv,s_ix) in zip(self.s_categories_,range(s_len)):
                        
                        # compute marginal
                        Ps = sum(self.Pys.loc[:,sv])
                        
                        # upper constraint
                        Gp[:,y_ix,s_ix] = ( (Ps_x[:,s_ix]/Ps) - (1+alpha) )
                        
                        # lower constraint
                        Gm[:,y_ix,s_ix] = ( -(Ps_x[:,s_ix]/Ps) + (1-alpha) )
                        
                    Glist.append(Gp)
                    Glist.append(Gm)

        G_temp = np.concatenate(Glist,axis=2)
        ################################################
        G_temp = G_temp + np.random.normal(loc=0.0, scale=1e-5, size=G_temp.shape)
        ################################################
        self.G = G_temp
        return G_temp
                

        
    
    def project(self,X,y=None,s=None,constraints=[('meo',.1)],rho=2,max_iter=1000,use_y = False,method='tf'):
        '''
        Project trained model
        '''
        assert self.Trained,  "Fit models first!"
        self.constraints = constraints
        
        # print('...Building constraint matrix...')
        G = self.buildG(X,self.constraints,y=y,s=s)
        
        
        # print('...Projecting...')
        
        fudge = 1e-4
        
        if not use_y:
            Py_x = self.clf_Y.predict_proba(X)
            
        else:
            Py_x = self.enc_y(fit_transform(y))
            
        Py_x = (Py_x+fudge)/((Py_x+fudge).sum(axis=1,keepdims=True))

        if method == 'tf':
            self.l = MP.admm_tf(G, np.expand_dims(Py_x,axis=2),rho=rho,max_iter=max_iter,div=self.div)
        elif method == 'np':
            self.l = MP.admm(G, np.expand_dims(Py_x, axis=2), rho=rho, max_iter=max_iter, report=True, div=self.div)
        else:
            print('Method can only be either tf or np!!!')
            return

        self.Projected = True
        
    
    def predict_proba(self,X,y=None,s=None):
        '''
        Predict with projected model.
        '''
        assert self.Projected, "Project model first!"
        
        fudge = 1e-4
        
        # print('...Building constraint matrix...')
        
        G = self.buildG(X,self.constraints,y=y,s=s)
        Py_x = self.clf_Y.predict_proba(X)
        
        Py_x = (Py_x+fudge)/((Py_x+fudge).sum(axis=1,keepdims=True))
        
        
        # print('...Predicting...')
        
        return MP.predict(self.l, G, np.expand_dims(Py_x,axis=2),div=self.div)
        
        
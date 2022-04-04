"""
The MBI Module
_
This module provides methods for performing multivariate Bayesian inversion
(MBI) for classification (MBC) and regression (MBI) based on Bayesian
estimation of multivariate general linear models (MGLM).

Author: Joram Soch, BCCN Berlin
E-Mail: joram.soch@bccn-berlin.de
Edited: 04/04/2022, 16:09
"""


# import packages
#-----------------------------------------------------------------------------#
import cvBMS
import numpy as np


###############################################################################
# class: multivariate general linear model                                    #
###############################################################################
class model:
    """
    Multivariate General Linear Model for MBI prediction
    """
    
    # initialize MGLM
    #-------------------------------------------------------------------------#
    def __init__(self, Y, x, X=None, V=None, mb_type='MBC'):
        """
        Initialize Model for Multivariate Bayesian Inversion
        """
        
        # create design matrix
        if mb_type == 'MBC':                # classification
            C  = int(np.max(x))
            Xx = np.zeros((Y.shape[0],C))
            for j in range(C):
                Xx[x==C,j+1] = 1
        elif mb_type == 'MBR':              # regression
            Xx = np.hstack((x, np.ones((Y.shape[0],1))))
        if X is None: X = np.zeros((Y.shape[0],0))
        if V is None: V = np.eye(Y.shape[0])
        
        # store model information
        X         = np.hstack((Xx, X))      # enhanced design matrix
        self      = cvBMS.MGLM(Y, X, V)     # multivariate GLM object
        self.x    = x
        if mb_type == 'MBC':
            self.is_MBC = True
        elif mb_type == 'MBR':
            self.is_MBC = False

    # function: training for MBI
    #-------------------------------------------------------------------------#
    def train(self):
        """
        Training for Multivariate Bayesian Inversion
        """
        
        # specify prior parameters
        M0 = np.zeros((self.p,self.v))
        L0 = np.zeros((self.p,self.p))
        O0 = np.zeros((self.p,self.p))
        v0 = 0
        
        # calculate posterior parameters
        M1, L1, O1, v1 = self.Bayes(M0, L0, O0, v0)
        
        # assemble MBA dictionary
        MBA = {
            "input": {"x": self.x},
            "data" : {"Y1": self.Y, "X1": self.X, "V1": self.V},
            "prior": {"M0": M0, "L0": L0, "O0": O0, "v0": v0},
            "post" : {"M1": M1, "L1": L1, "O1": O1, "v1": v1}
        }
        return MBA
    
    # function: testing for MBI
    #-------------------------------------------------------------------------#
    def test(self, MBA, prior=None):
        """
        Testing for Multivariate Bayesian Inversion
        """
        
        # set prior if required
        if prior is None:
            if self.is_MBC:
                C = int(np.max(MBA['input']['x']))
                prior.x = np.arange(0,C) + 1
                prior.p = (1/C) * np.ones(C)
            else:
                L = 100
                prior.x = np.linspace(np.min(MBA['input']['x']), np.max(MBA['input']['x']), L)
                prior.p = (1/(np.max(MBA['input']['x'])-np.min(MBA['input']['x']))) * np.ones(L)
        
        # specify prior parameters
        M1 = MBA['post']['M1']
        L1 = MBA['post']['L1']
        O1 = MBA['post']['O1']
        v1 = MBA['post']['v1']
        
        # calculate posterior probabilities
        L     = prior.x.size
        PP    = np.zeros(self.n,L)
        logPP = np.zeros(self.n,L)
        for i in range(self.n):
            y2i = self.Y[i,:]
            x2i = self.X[i,:]
            vii = self.V[i,i]
            for j in range(L):
                x2ij = x2i
                if self.is_MBC:
                    x2ij[0,:L] = 0
                    x2ij[j]    = 1
                else:
                    x2ij[1]    = prior.x[j]
                mij            = cvBMS.MGLM(y2i, x2ij, vii)
                M2, L2, O2, v2 = mij.Bayes(M1, L1, O1, v1)
                logPP[i,j] = -self.v/2 * np.log(np.linalg.det(L2)) \
                           -      v2/2 * np.log(np.linalg.det(O2)) \
                           +             np.log(prior.p[j])
            PP[i,:] = np.exp(logPP[i,:] - np.mean(logPP[i,:]))
            if self.is_MBC:
                PP[i,:] = (1/np.sum(PP[i,:])) * PP[i,:]
            else:
                PP[i,:] = (1/np.trapz(prior.x, PP[i,:])) * PP[i,:]
        return PP


###############################################################################
# class: cross-validated MBI prediction                                       #
###############################################################################
class cvMBI:
    """
    Cross-Validated Multivariate Bayesian Classification or Regression
    """
    
    # initialize cvMBI
    #-------------------------------------------------------------------------#
    def __init__(self, Y, x, X=None, V=None, mb_type='MBC'):
        """
        Initialize Multivariate Bayesian Inversion with Cross-Validation
        """
        
        # set matrices if required
        if X is None: X = np.zeros((Y.shape[0],0))
        if V is None: V = np.eye(Y.shape[0])

        # store model information
        if mb_type == 'MBC':
            self.is_MBC = True
        elif mb_type == 'MBR':
            self.is_MBC = False
        self.mb_type = mb_type
        self.Y = Y
        self.x = x
        self.X = X
        self.V = V
        self.n = Y.shape[0]
        self.v = Y.shape[1]
        
    # function: cross-validation
    #-------------------------------------------------------------------------#
    def crossval(self, c=None, k=10, cv_mode='kfc'):
        """
        Cross-Validate for Multivariate Bayesian Inversion
        """
        
        # set defaults values
        if c is None:
            if self.is_MBC: c = self.x
            else:           c = np.ones(self.x.size)
        if np.max(c) == 1:
            if cv_mode == 'kfc' : cv_mode = 'kf'
            if cv_mode == 'looc': cv_mode = 'loo'
        if cv_mode == 'loo': k = c.size
        
        # get class indices
        n  = c.size
        C  = np.max(c)
        ic = [None] * C
        nc = np.zeros(C)
        for j in range(C):
            ic[j] = np.nonzero(c==j+1)
            nc[j] = c[ic[j]].size
        CV = np.zeros((n,k))
        
        # k-folds and leave-one-out cross-validation
        if cv_mode == 'kf' or cv_mode == 'loo':
            nf = np.ceil(n/k)
            ia = np.arange(0,n)
            for g in range(k):
                i2 = np.arange(g*nf, np.min((g+1)*nf, n))
                i1 = [i for i in ia if i not in i2]
                CV[i1,g] = 1
                CV[i2,g] = 2
        
        # cross-validation points per class
        if cv_mode == 'kfc' or cv_mode == 'looc':
            nf = np.ceil(nc/k)
            ia = np.arange(0,n)
            for g in range (k):
                i2 = np.empty(1)
                for j in range(C):
                    i2.append(ic[j][np.arange(g*nf[j], np.min((g+1)*nf[j], nc[j]))])
                i1 = [i for i in ia if i not in i2]
            CV[i1,g] = 1
            CV[i2,g] = 2
        
        # store CV information
        self.CV = CV
        
    # function: MBI-based prediction
    #-------------------------------------------------------------------------#
    def predict(self, prior=None):
        """
        Predict based on Multivariate Bayesian Inversion
        """
        
        # set CV if required
        if not hasattr(self, 'CV'):
            self.crossval()
        k = self.CV.shape[1]
        
        # set prior if required
        if prior is None:
            if self.is_MBC:
                C = int(np.max(self.x))
                prior.x = np.arange(0,C) + 1
                prior.p = (1/C) * np.ones(C)
            else:
                L = 100
                prior.x = np.linspace(np.min(self.x), np.max(self.x), L)
                prior.p = (1/(np.max(self.x)-np.min(self.x))) * np.ones(L)
        self.prior = prior
        
        # cross-validated analysis
        L  = prior.x.size                   # classes/levels
        xt = np.zeros(self.n)               # "true" classes
        xp = np.zeros(self.n)               # predicted classes
        PP = np.zeros((self.n,L))           # posterior probabilities
        for g in range(k):
            
            # get test and training set
            i1 = np.nonzero(self.CV[:,g]==1)
            i2 = np.nonzero(self.CV[:,g]==2)
            Y1 = self.Y[i1,:]
            Y2 = self.Y[i2,:]
            x1 = self.x[i1]
            x2 = self.x[i2]
            if self.X.shape[2] > 0:
                X1 = self.X[i1,:]
                X2 = self.X[i2,:]
            else:
                X1 = None
                X2 = None
            V1 = self.V[i1,:][:,i1]
            V2 = self.V[i2,:][:,i2]
            
            # training data: X is known, infer on B/T
            m1   = model(Y1, x1, X1, V1, self.mb_type)
            MBA1 = m1.train()
            
            # test data: B/T are known, infer on X
            m2   = model(Y2, x2, X2, V2, self.mb_type)
            PP2  = m2.test(MBA1, prior)
            PP[i2,:] = PP2
            
            # collect true and predicted
            for i in range(i2):
                xt[i2[i]] = x2[i]
                xp[i2[i]] = prior.x[PP[i2[i]]==np.max(PP[i2[i]])]
        
        # store prediction results
        self.xt = xt
        self.xp = xp
        self.PP = PP
        
    # function: MBI-based prediction
    #-------------------------------------------------------------------------#
    def evaluate(self, meas=None):
        """
        Evaluate Performance of Multivariate Bayesian Inversion
        """
        
        # set measure if required
        if meas is None:
            if self.is_MBC: meas = 'DA'
            if self.is_MBC: meas = 'r'
        
        # calculate performance
        if meas == 'DA':                    # classification
            perf = np.mean(self.xp==self.xt)
        if meas == 'r':                     # regression
            R    = np.corrcoef(self.xp, self.xt)
            perf = R[0,1]
        
        # return performance
        return perf
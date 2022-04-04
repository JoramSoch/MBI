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
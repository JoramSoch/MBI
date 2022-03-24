"""
The cvBMS Unit
_
This module assembles methods for calculating the cross-validated log model
evidence (cvLME) for different sorts of statistical models and to perform
cross-validated Bayesian model seelection (cvBMS) based on these cvLMEs.

The general principles of the cvBMS unit are as follows:
- To use the module, it is simply imported via
      import cvBMS ,
  e.g. at the beginning of the analysis script.
- First, a model object is generated using a command like
      model = cvBMS.<name-of-the-model-class>(Y, [X])
  where Y are measured data and X are independent variables, if any.
- Second, the cvLME method is applied using a command like
      cvLME_model = model.cvLME(S)
  where S is the number of subsets into which the data are partitioned.
- Generally, this module applies mass-univariate analysis, i.e. columns of
  the data matrix are treated as different measurements which are analyzed
  separately, but using the same model.

So far, the following model structures are available:
- MS    = model space; for general model selection operations
- GLM   = univariate general linear model; for linear regression
- Poiss = Poisson distribution with exposures; for count data

Author: Joram Soch, BCCN Berlin
E-Mail: joram.soch@bccn-berlin.de
Edited: 21/02/2019, 14:50
"""


# import packages
#-----------------------------------------------------------------------------#
import numpy as np
import scipy.special as sp_special


###############################################################################
# class: model space                                                          #
###############################################################################
class MS:
    """
    The model space class allows to perform model comparison, model selection
    and family inference over a space of candidate models. A model space is
    defined by a number of log model evidences (LMEs) from which measures
    such as log Bayes factors (LBFs), posterior model probabilities (PPs)
    and log family evidences (LFEs) can be derived.
    
    Edited: 01/02/2019, 12:30
    """
    
    # initialize MS
    #-------------------------------------------------------------------------#
    def __init__(self, LME):
        """
        Initialize a Model Space
        ms = cvBMS.MS(LME)
            LME   - an M x N array of LMEs
            ms    - a model space object
            o LME - the Mx N array of log model evidences
            o M   - the number of models in the model space
            o N   - the number of instances to be analyzed
        """
        self.LME = LME          # log model evidences
        self.M   = LME.shape[0] # number of models
        self.N   = LME.shape[1] # number of instances
        
    # function: log Bayes factor
    #-------------------------------------------------------------------------#
    def LBF(self, m1=1, m2=2):
        """
        Return Log Bayes Factor between Two Models
        LBF12 = ms.LBF(m1, m2)
            m1    - index of the first model to be compared (default: 1)
            m2    - index of the second model to be compared (default: 2)
            LBF12 - a 1 x N vector of log Bayes factors in favor of first model
        """
        LBF12 = self.LME[m1-1,:] - self.LME[m2-1,:]
        return LBF12
    
    # fucntion: Bayes factor
    #-------------------------------------------------------------------------#
    def BF(self, m1=1, m2=2):
        """
        Return Bayes Factor between Two Models
        BF12 = ms.BF(m1, m2)
            m1   - index of the first model to be compared (default: 1)
            m2   - index of the second model to be compared (default: 2)
            BF12 - a 1 x N vector of Bayes factors in favor of first model
        """
        BF12 = np.exp(self.LBF(m1, m2))
        return BF12
        
    # function: posterior model probabilities
    #-------------------------------------------------------------------------#
    def PP(self, prior=None):
        """
        Return Posterior Probabilities of Several Models
        post = ms.PP(prior)
            prior - an M x 1 vector
                    or M x N matrix of prior model probabilities (optional)
            post  - an M x N matrix of posterior model probabilities
        """
        
        # set uniform prior
        if prior is None:
            prior = 1/self.M * np.ones((self.M,1))
        if prior.shape[1] == 1:
            prior = np.tile(prior, (1, self.N))
            
        # subtract average LMEs
        LME = self.LME
        LME = LME - np.tile(np.mean(LME,0), (self.M, 1))
        
        # calculate PPs
        post = np.exp(LME) * prior
        post = post / np.tile(np.sum(post,0), (self.M, 1))
        
        # return PPs
        return post
    
    # function: log family evidences
    #-------------------------------------------------------------------------#
    def LFE(self, m2f):
        """
        Return Log Family Evidences for Model Families
        LFE = ms.LFE(m2f)
            m2f - a 1 x M vector specifying family affiliation, i.e.
                  m2f(i) = f -> i-th model belongs to j-th family
        """
                
        # get number of model families
        F = np.int(m2f.max())
        
        # calculate log family evidences
        #---------------------------------------------------------------------#
        LFE = np.zeros((F,self.N))
        for f in range(F):
            
            # get models from family
            mf = [i for i, m in enumerate(m2f) if m == (f+1)]
            Mf = len(mf)
            
            # set uniform prior
            prior = 1/Mf * np.ones((Mf,1))
            prior = np.tile(prior, (1, self.N))
            
            # calculate LFEs
            LME_fam  = self.LME[mf,:]
            LME_fam  = LME_fam + np.log(prior) + np.log(Mf)
            LME_max  = LME_fam.max(0)
            LME_diff = LME_fam - np.tile(LME_max, (Mf, 1))
            LFE[f,:] = LME_max + np.log(np.mean(np.exp(LME_diff),0))
            
        # return log family evidence
        return LFE


###############################################################################
# class: univariate general linear model                                      #
###############################################################################
class GLM:
    """
    The GLM class allows to specify, estimate and assess univariate general
    linear models a.k.a. linear regression which is defined by an n x 1 data
    vector y, an n x p design matrix X and an n x n covariance matrix V.
    
    Edited: 01/02/2019, 08:30
    """
    
    # initialize GLM
    #-------------------------------------------------------------------------#
    def __init__(self, Y, X, V=None):
        """
        Initialize a General Linear Model
        glm = cvBMS.GLM(Y, X, V)
            Y   - an n x v data matrix of measured signals
            X   - an n x p design matrix of predictor variables
            V   - an n x n covariance matrix specifying correlations (default: I_n)
            glm - a GLM object
            o Y - the n x v data matrix
            o X - the n x p design matrix
            o V - the n x n covariance matrix
            o n - the number of observations
            o v - the number of instances
            o p - the number of regressors
        """
        self.Y = Y                          # data matrix
        self.X = X                          # design matrix
        if V is None: V = np.eye(Y.shape[0])# covariance matrix
        self.V = V                          # covariance matrix
        self.P = np.linalg.inv(V)           # precision matrix
        self.n = Y.shape[0]                 # number of observations
        self.v = Y.shape[1]                 # number of instances
        self.p = X.shape[1]                 # number of regressors
        
    # function: ordinary least squares
    #-------------------------------------------------------------------------#
    def OLS(self):
        """
        Ordinary Least Squares for General Linear Model
        B_est = glm.OLS()
            B_est - a p x v matrix of estimated regression coefficients
        """
        B_est = np.linalg.inv(self.X.T @ self.X) @ (self.X.T @ self.Y)
        return B_est
    
    # function: weighted least squares
    #-------------------------------------------------------------------------#
    def WLS(self):
        """
        Weighted Least Squares for General Linear Model
        B_est = glm.WLS()
            B_est - a p x v matrix of estimated regression coefficients
        """
        B_cov = np.linalg.inv(self.X.T @ self.P @ self.X)
        B_est = B_cov @ (self.X.T @ self.P @ self.Y)
        return B_est
    
    # function: maximum likelihood estimation
    #-------------------------------------------------------------------------#
    def MLE(self):
        """
        Maximum Likelihood Estimation for General Linear Model
        (B_est, s2_est) = glm.MLE()
            B_est  - a p x v matrix of estimated regression coefficients
            s2_est - a 1 x v vector of estimated residual variances
        """
        B_est  = self.WLS()
        E_est  = self.Y - (self.X @ B_est)
        s2_est = np.zeros(self.v)
        for j in range(self.v):
            s2_est[j] = 1/self.n * (E_est[:,j].T @ self.P @ E_est[:,j])
        return B_est, s2_est
    
    # function: Bayesian estimation
    #-------------------------------------------------------------------------#
    def Bayes(self, m0, L0, a0, b0):
        """
        Bayesian Estimation of General Linear Model with Normal-Gamma Priors
        (mn, Ln, an, bn) = glm.Bayes(m0, L0, a0, b0)
            m0 - a p x v vector (prior means of regression coefficients)
            L0 - a p x p matrix (prior precision of regression coefficients)
            a0 - a 1 x 1 scalar (prior shape of residual precision)
            b0 - a 1 x v vector (prior rates of residual precision)
            mn - a p x v vector (posterior means of regression coefficients)
            Ln - a p x p matrix (posterior precision of regression coefficients)
            an - a 1 x 1 scalar (posterior shape of residual precision)
            bn - a 1 x v vector (posterior rates of residual precision)
        """
        
        # enlarge priors if required
        if m0.shape[1] == 1:
            m0 = np.tile(m0, (1, self.v))
        if np.isscalar(b0):
            b0 = b0 * np.ones(self.v)
        
        # estimate posterior parameters
        Ln = self.X.T @ self.P @ self.X + L0
        mn = np.linalg.inv(Ln) @ ( self.X.T @ self.P @ self.Y + L0 @ m0 )
        an = a0 + self.n/2
        bn = np.zeros(self.v)
        for j in range(self.v):
            bn[j] = b0[j] + 1/2 * ( self.Y[:,j].T @ self.P @ self.Y[:,j] + m0[:,j].T @ L0 @ m0[:,j] - mn[:,j].T @ Ln @ mn[:,j] )
        
        # return posterior parameters
        return mn, Ln, an, bn
    
    # function: log model evidence
    #-------------------------------------------------------------------------#
    def LME(self, L0, a0, b0, Ln, an, bn):
        """
        Log Model Evidence of General Linear Model with Normal-Gamma Priors
        LME = glm.LME(L0, a0, b0, Ln, an, bn)
            L0  - a p x p matrix (prior precision of regression coefficients)
            a0  - a 1 x 1 scalar (prior shape of residual precision)
            b0  - a 1 x v vector (prior rate of residual precision)
            Ln  - a p x p matrix (posterior precision of regression coefficients)
            an  - a 1 x 1 scalar (posterior shape of residual precision)
            bn  - a 1 x v vector (posterior rate of residual precision)
            LME - a 1 x v vector of log model evidences
        """
        
        # calculate log model evidence
        LME = 1/2 * np.log(np.linalg.det(self.P)) - self.n/2 * np.log(2*np.pi)      \
            + 1/2 * np.log(np.linalg.det(L0))     - 1/2 * np.log(np.linalg.det(Ln)) \
            + sp_special.gammaln(an)              - sp_special.gammaln(a0)          \
            + a0 * np.log(b0)                     - an * np.log(bn)
        
        # return log model evidence
        return LME
    
    # function: cross-validated log model evidence
    #-------------------------------------------------------------------------#
    def cvLME(self, S=2):
        """
        Cross-Validated Log Model Evidence for General Linear Model
        cvLME = glm.cvLME(S)
            S     - the number of subsets into which data are partitioned (default: 2)
            cvLME - a 1 x v vector of cross-validated log model evidences
        """
        
        # determine data partition
        npS  = np.int(self.n/S);# number of data points per subset, truncated
        inds = range(S*npS)     # indices for all data, without remainders
        
        # set non-informative priors
        m0_ni = np.zeros((self.p,1))        # flat Gaussian
        L0_ni = np.zeros((self.p,self.p))
        a0_ni = 0;                          # Jeffrey's prior
        b0_ni = 0;
        
        # calculate out-of-sample log model evidences
        #---------------------------------------------------------------------#
        oosLME = np.zeros((S,self.v))
        for j in range(S):
            
            # set indices
            i2 = range(j*npS, (j+1)*npS)                # test indices
            i1 = [i for i in inds if i not in i2]       # training indices
            
            # partition data
            Y1 = self.Y[i1,:]                           # training data
            X1 = self.X[i1,:]
            P1 = self.P[i1,:][:,i1]
            S1 = GLM(Y1, X1, P1)
            Y2 = self.Y[i2,:]                           # test data
            X2 = self.X[i2,:]
            P2 = self.P[i2,:][:,i2]
            S2 = GLM(Y2, X2, P2)
            
            # calculate oosLME
            m01 = m0_ni; L01 = L0_ni; a01 = a0_ni; b01 = b0_ni;
            mn1, Ln1, an1, bn1 = S1.Bayes(m01, L01, a01, b01)
            m02 = mn1; L02 = Ln1; a02 = an1; b02 = bn1;
            mn2, Ln2, an2, bn2 = S2.Bayes(m02, L02, a02, b02)
            oosLME[j,:] = S2.LME(L02, a02, b02, Ln2, an2, bn2)
            
        # return cross-validated log model evidence
        cvLME = np.sum(oosLME,0)
        return cvLME
    
    
###############################################################################
# class: Poisson distribution                                                 #
###############################################################################
class Poiss:
    """
    The Poisson class allows to specify, estimate and assess basic Poisson
    models which are defined by an n x 1 data vector y and an n x 1 design
    vector of exposure values x.
    
    Edited: 13/02/2019, 07:25
    """
    
    # initialize Poisson
    #-------------------------------------------------------------------------#
    def __init__(self, Y, x=None):
        """
        Initialize a Poisson Distribution
        poiss = cvBMS.Poiss(Y, x)
            Y     - an n x v data matrix of measured counts
            x     - an n x 1 design vector of exposure values (default: 1_n)
            poiss - a Poisson object
            o Y - the n x v data matrix
            o x - the n x 1 design vector
            o n - the number of observations
            o v - the number of instances
        """
        self.Y = Y                          # data matrix
        if x is None:
            x = np.ones(Y.shape[0])         # design vector
        self.x = x
        self.n = Y.shape[0]                 # number of observations
        self.v = Y.shape[1]                 # number of instances
        
    # function: maximum likelihood estimation
    #-------------------------------------------------------------------------#
    def MLE(self):
        """
        Maximum Likelihood Estimation for Poisson Distribution
        l_est = poiss.MLE()
            l_est - a 1 x v vector of estimated Poisson rates
        """
        l_est = np.sum(self.Y,0)/np.sum(self.x)
        return l_est
    
    # function: Bayesian estimation
    #-------------------------------------------------------------------------#
    def Bayes(self, a0, b0):
        """
        Bayesian Estimation of Poisson Distribution with Gamma Prior
        (an, bn) = poiss.Bayes(a0, b0)
            a0 - a 1 x v vector (prior shapes of the Poisson rates)
            b0 - a 1 x 1 scalar (prior rate of the Poisson rates)
            an - a 1 x v scalar (posterior shapes of the Poisson rates)
            bn - a 1 x 1 vector (posterior rate of the Poisson rates)
        """
        
        # enlarge priors if required
        if np.isscalar(a0):
            a0 = a0 * np.ones(self.v)
        
        # estimate posterior parameters
        an = a0 + self.n * np.sum(self.Y,0)
        bn = b0 + self.n * np.sum(self.x)
        
        # return posterior parameters
        return an, bn
    
    # function: log model evidence
    #-------------------------------------------------------------------------#
    def LME(self, a0, b0, an, bn):
        """
        Log Model Evidence of Poisson Distribution with Gamma Prior
        LME = poiss.LME(a0, b0, an, bn)
            a0  - a 1 x v vector (prior shapes of the Poisson rates)
            b0  - a 1 x 1 scalar (prior rate of the Poisson rates)
            an  - a 1 x v scalar (posterior shapes of the Poisson rates)
            bn  - a 1 x 1 vector (posterior rate of the Poisson rates)
            LME - a 1 x v vector of log model evidences
        """
        
        # calculate log model evidence
        x   = np.reshape(self.x, (self.n, 1))
        X   = np.tile(x, (1, self.v))
        LME = np.sum(self.Y * np.log(X), 0) - np.sum(sp_special.gammaln(self.Y+1), 0) \
            + sp_special.gammaln(an)        - sp_special.gammaln(a0)                  \
            + a0 * np.log(b0)               - an * np.log(bn)
        
        # return log model evidence
        return LME
    
    # function: cross-validated log model evidence
    #-------------------------------------------------------------------------#
    def cvLME(self, S=2):
        """
        Cross-Validated Log Model Evidence for Poisson Distribution
        cvLME = poiss.cvLME(S)
            S     - the number of subsets into which data are partitioned (default: 2)
            cvLME - a 1 x v vector of cross-validated log model evidences
        """
        
        # determine data partition
        npS  = np.int(self.n/S);# number of data points per subset, truncated
        inds = range(S*npS)     # indices for all data, without remainders
        
        # set non-informative priors
        a0_ni = 0;
        b0_ni = 0;
        
        # calculate out-of-sample log model evidences
        #---------------------------------------------------------------------#
        oosLME = np.zeros((S,self.v))
        for j in range(S):
            
            # set indices
            i2 = range(j*npS, (j+1)*npS)                # test indices
            i1 = [i for i in inds if i not in i2]       # training indices
            
            # partition data
            Y1 = self.Y[i1,:]                           # training data
            x1 = self.x[i1]
            S1 = Poiss(Y1, x1)
            Y2 = self.Y[i2,:]                           # test data
            x2 = self.x[i2]
            S2 = Poiss(Y2, x2)
            
            # calculate oosLME
            a01 = a0_ni; b01 = b0_ni;
            an1, bn1 = S1.Bayes(a01, b01)
            a02 = an1; b02 = bn1;
            an2, bn2 = S2.Bayes(a02, b02)
            oosLME[j,:] = S2.LME(a02, b02, an2, bn2)
            
        # return cross-validated log model evidence
        cvLME = np.sum(oosLME,0)
        return cvLME
    
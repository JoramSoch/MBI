"""
The MBI Module
_
This module provides methods for performing multivariate Bayesian inversion
(MBI) for classification (MBC) and regression (MBI) based on Bayesian
estimation of multivariate general linear models (MGLM).

The module contains two classes:
- model (a sub-class inherited from cvBMS.MGLM):
      m = MBI.model(Y, x, [...], mb_type)
  creates a multivariate general linear model (MGLM) object using
  the labels x as a predictor variable for the features Y and prepares a
  classification (mb_type='MBC') or regression (mb_type='MBR') analysis.
  The class allows for the following methods:
  - train: calibrates model parameters using Bayesian MGLM estimation
  - test : predicts labels using estimated parameters and MGLM inversion
- cvMBI (a unique class from this module):
      M = MBI.cvMBI(Y, x, [...], mb_type)
  creates a cross-validated multivariate Bayesian inversion (cvMBI) object
  with features Y, labels x and analysis type mb_type (see above).
  The class allows for the following methods:
  - crossval: defines the cross-validation scheme for the MBI analysis
  - predict : performs the MBI analysis using model.train and model.test
  - evaluate: evaluates performance of the predictive model
  
Additionally, the module contains two functions:
- matnrnd : samples from the matrix-variate normal distribution
- uniprior: creates mass or density function for a uniform prior

For more information, see the usage examples in the readme file:
  https://github.com/JoramSoch/MBI/blob/main/README.md

Author: Joram Soch, BCCN Berlin
E-Mail: joram.soch@bccn-berlin.de
Edited: 28/02/2025, 10:14
"""


# import packages
#-----------------------------------------------------------------------------#
import cvBMS
import numpy as np


# function: matrix-normal random numbers
#-----------------------------------------------------------------------------#
def matnrnd(M, U, V, c=1, A=None, B=None):
    """
    Random Matrices from the Matrix-Variate Normal Distribution
    R = matnrnd(M, U, V, c, A, B)
        M - an n x p matrix, the mean of the matrix normal distribution
        U - an n x n matrix, the covariance across rows of the matrix
        V - a  p x p matrix, the covariance across columns of the matrix
        c - an integer, the number of cases to be drawn from the distribution (default: 1)
        A - an n x n matrix, the upper Cholesky decomposition of U (optional)
        B - a  p x p matrix, the lower Cholesky decomposition of V (optional)
    
        R - an n x p x c array of random matrices from the distribution
    """
    
    # set default values
    if c is None:
        c = 1
    if A is None:
        A = np.linalg.cholesky(U)
    if B is None:
        B = np.linalg.cholesky(V).T
    # Note: Python's numpy.linalg.cholesky seems to return the transpose
    # relative to MATLAB's chol. This is why transposition in this Python
    # code is reversed in comparison with the MATLAB implementation:
    #   matnrnd.m           MBI.matnrnd
    #   ---------           -----------
    #   A = chol(U)';       A = np.linalg.cholesky(U)
    #   B = chol(V);        B = np.linalg.cholesky(V).T
    
    # sample from standard normal distribution
    R = np.random.standard_normal(size=(U.shape[0], V.shape[0], c))
    
    # transform into matrix normal distribution
    for i in range(c):
        R[:,:,i] = M + A @ R[:,:,i] @ B
    
    # return random matrix
    return R

# function: uniform prior distribution
#-----------------------------------------------------------------------------#
def uniprior(x_type='disc', L=100, x_min=0, x_max=1):
    """
    Mass or Density Function for a Uniform Prior Distribution
    prior = uniprior(x_type, L, x_min, x_max)
        x_type - a string indicating the type of random variable (default: 'disc')
                 o 'disc' - discrete random variable
                 o 'cont' - continuous random variable
        L      - an integer, the number of possible values (default: 100)
        x_min  - a scalar, the minimum possible value (only if x_type is 'cont')
        x_max  - a scalar, the maximum possible value (only if x_type is 'cont')
        
        prior  - a dictionary specifying the prior distribution
        o x   - a  1 x L vector of possible label values
        o p   - a  1 x L vector of probabilities or densities 
    """
    
    # discrete random variable (classes)
    if x_type == 'disc':
        C = int(L)
        prior = {
            "x": np.arange(0,C) + 1,
            "p": (1/C) * np.ones(C)
        }
    
    # continuous random variable (targets)
    if x_type == 'cont':
        if L is None: L = 100
        prior = {
            "x": np.linspace(x_min, x_max, L),
            "p": (1/(x_max-x_min)) * np.ones(L)
        }
    
    # return uniform prior
    return prior


###############################################################################
# class: multivariate general linear model                                    #
###############################################################################
class model(cvBMS.MGLM):
    """
    Multivariate General Linear Model for MBI prediction (class)
    """
    
    # initialize MGLM
    #-------------------------------------------------------------------------#
    def __init__(self, Y, x, X=None, V=None, mb_type='MBC'):
        """
        Initialize Model for Multivariate Bayesian Inversion
        mglm = MBI.model(Y, x, X, V, mb_type)
            Y       - an n x v data matrix of feature variables
            x       - an n x 1 label vector of class indices (MBC) or
                      an n x 1 label vector of target values (MBR)
            X       - an n x r covariate matrix (default: n x 0 empty matrix)
            V       - an n x n covariance matrix (default: n x n identity matrix)
            mb_type - a string specifying the analysis type (default: 'MBC')
                      o 'MBC' - multivariate Bayesian classification
                      o 'MBR' - multivariate Bayesian regression

            mglm    - an MGLM object (see cvBMS.MGLM)
            o Y     - the n x v data matrix
            o X     - the n x p design matrix (created from labels x and covariates X)
            o V     - the n x n covariance matrix
            o x     - the n x 1 label vector
            o is_MBC- a logical indicating the analysis type
                      o True  - MBC
                      o False - MBR
            
        The design matrix mglm.X is generated by either creating C indicator
        regressors (containing only 0s and 1s) according to the class indices
        x (where C is the number of classes), if classification (MBC), or by
        concatenating the the target values x with an n x 1 vector of ones,
        if regression (MBR). If the input X is supplied, these covariates are
        added as additional regressors to the final design matrix mglm.X.
        """
        
        # create design matrix
        if mb_type == 'MBC':                # classification
            n  = Y.shape[0]
            C  = int(np.max(x))
            Xx = np.zeros((n,C))
            for i in range(n):
                Xx[i,int(x[i]-1)] = 1
        elif mb_type == 'MBR':              # regression
            Xx = np.c_[x, np.ones((Y.shape[0],1))]
        if X is None: X = np.zeros((Y.shape[0],0))
        if V is None: V = np.eye(Y.shape[0])
        
        # store model information
        X = np.c_[Xx, X]                    # enhanced design matrix
        super().__init__(Y, X, V)           # inherit parent class
        self.x      = x
        self.is_MBC =(mb_type == 'MBC')
    
    # function: training for MBI
    #-------------------------------------------------------------------------#
    def train(self):
        """
        Training for Multivariate Bayesian Inversion
        MBA = mglm.train()
            MBA    - a dictionary, the trained multivariate Bayesian automaton
            o input: user input
              o x: labels used for specifying the model
            o data : training data
              o Y1, X1, V1: see MBI.model.__init__
            o prior: non-informative prior parameters (before training)
              o M0, L0, O0, v0: see cvBMS.MGLM.Bayes
            o post : informative posterior parameters (after training)
              o Mn, Ln, On, vn: see cvBMS.MGLM.Bayes
        """
        
        # specify prior parameters
        M0 = np.zeros((self.p,self.v))
        L0 = np.zeros((self.p,self.p))
        O0 = np.zeros((self.v,self.v))
        v0 = 0
        
        # calculate posterior parameters
        M1, L1, O1, v1 = self.Bayes(M0, L0, O0, v0)
        
        # assemble MBA dictionary
        MBA = {
            "input": {"x" : self.x},
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
        PP = mglm.test(MBA, prior)
            MBA   - a dictionary specifying a trained MBI model (see MBI.model.train)
            prior - a dictionary specifying the prior distribution
            o x   - a  1 x L vector, the support of the distribution
            o p   - a  1 x L vector, the probability mass or density
             (L   - number of classes or number of target variable levels)
            
            PP    - an n x L matrix giving the posterior probabilities
            
        Note that MBA has to be obtained from a different set of training data.
        
        If prior is not supplied, it will be specified as a uniform prior
        distribution (see MBI.uniprior), i.e. as a discrete uniform distribution
        over classes (thus: L = C), if classification (MBC), or as a continuous
        uniform distribution with L levels (default: L = 100) over the range
        between the minimum and the maximum of value of x in the training data,
        if regression (MBR).
        """
        
        # set prior if required
        if prior is None:
            if self.is_MBC:
                prior = uniprior('disc', int(np.max(MBA['input']['x'])))
            else:
                prior = uniprior('cont', 100, np.min(MBA['input']['x']), np.max(MBA['input']['x']))
        
        # specify prior parameters
        M1 = MBA['post']['M1']
        L1 = MBA['post']['L1']
        O1 = MBA['post']['O1']
        v1 = MBA['post']['v1']
        
        # calculate posterior probabilities
        L     = prior['x'].size
        PP    = np.zeros((self.n,L))
        logPP = np.zeros((self.n,L))
        p_ind = [j for j in range(L) if prior['p'][j] != 0]
        
        # for each data point in the test set
        for i in range(self.n):
            y2i = np.array([ self.Y[i,:] ])
            x2i = np.array([ self.X[i,:] ])
            vii = np.array([[self.V.diagonal()[i]]])
            
            # for each label value (where prior is non-zero)
            for j in p_ind:  
                x2ij = x2i
                if self.is_MBC:             # classification -> categorical
                    x2ij[0,:L] = 0
                    x2ij[0,j]  = 1
                else:                       # regression -> parametric
                    x2ij[0,0]  = prior['x'][j]
                # specify model assuming that the label = x for this point
                mij            = cvBMS.MGLM(y2i, x2ij, vii)
                M2, L2, O2, v2 = mij.Bayes(M1, L1, O1, v1)
                # calculate posterior probability of label = x, given data
                logPP[i,j] = -self.v/2 * np.linalg.slogdet(L2)[1] \
                           -      v2/2 * np.linalg.slogdet(O2)[1] \
                           +             np.log(prior['p'][j])
              # PP[i,j]    = np.sqrt( np.power(np.linalg.det(L2), -v) /
              #                       np.power(np.linalg.det(O2), v2))*
              #              prior['p'][j]
            
            PP[i,p_ind] = np.exp(logPP[i,p_ind] - np.mean(logPP[i,p_ind]))
          # PP[i,:]     = np.exp(logPP[i,:] - np.mean(logPP[i,:]))
            # normalize posterior distribution to sum/integrate to one
            if self.is_MBC:                 # classification -> mass
                PP[i,:] = (1/np.sum(PP[i,:])) * PP[i,:]
            else:                           # regression -> density
                PP[i,:] = (1/np.trapz(PP[i,:], prior['x'])) * PP[i,:]
        
        # return posterior probabilities
        return PP


###############################################################################
# class: cross-validated MBI prediction                                       #
###############################################################################
class cvMBI:
    """
    Cross-Validated Multivariate Bayesian Inversion (class)
    """
    
    # initialize cvMBI
    #-------------------------------------------------------------------------#
    def __init__(self, Y, x, X=None, V=None, mb_type='MBC'):
        """
        Initialize Multivariate Bayesian Inversion with Cross-Validation
        MBCR = MBI.cvMBI(Y, x, X, V, mb_type)
            Y       - an n x v data matrix of feature variables
            x       - an n x 1 label vector of class indices (MBC) or
                      an n x 1 label vector of target values (MBR)
            X       - an n x r covariate matrix (default: n x 0 empty matrix)
            V       - an n x n covariance matrix (default: n x n identity matrix)
            mb_type - a string specifying the analysis type (default: 'MBC')
                      o 'MBC' - multivariate Bayesian classification
                      o 'MBR' - multivariate Bayesian regression

            MBCR    - an object for multivariate Bayesian classification or regression
            o Y     - the n x v data matrix
            o x     - the n x 1 label vector
            o X     - the n x r covariate matrix
            o V     - the n x n covariance matrix
            o is_MBC- a logical indicating the analysis type
                      o True  - MBC
                      o False - MBR
        
        Initializing merely stores features Y, labels x, covariates X and
        covariance V into an object for later furter operations (see below).
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
    def crossval(self, c=None, k=10, cv_mode='kfc', cv_mat=None):
        """
        Cross-Validate for Multivariate Bayesian Inversion
        MBCR.crossval(c, k, cv_mode, cv_mat)
            c       - an n x 1 vector of class indices
            k       - an integer, the number of cross-validation folds (default: 10)
            cv_mode - a string specifying the cross-validation mode (default: 'kfc')
                      o 'kf'     - k-folds cross-validation across all points
                      o 'kfc'    - k-folds cross-validation on points per class
                      o 'loo'    - leave-one-out cross-validation across all points
                      o 'looc'   - leave-one-out cross-validation on points per class
                      o 'custom' - custom cross-validation specified by n x k matrix
            cv_mat  - an n x k matrix specifying custom cross-validation (optional)
            
            MBCR    - an object for multivariate Bayesian classification or regression
            o CV    - an n x k matrix indicating training (1) and test (2)
                      data points for each cross-validation fold
            
        Notes:
        - If c is not supplied,
          - it is set to MBCR.x, if classification (MBC), or
          - specified as all ones, if regression (MBR).
        - If c contains all ones,
          - 'kfc' is automatically changed to 'kf' and
          - 'looc' is automatically changed to 'loo'.
        - If cv_mode is 'loo', k is automatically set to n.
        - If cv_mode is 'custom', CV is autometically set to cv_mat.
        """
        
        # set defaults values
        if c is None:
            if self.is_MBC: c = self.x
            else:           c = np.ones(self.x.size)
        if np.max(c) == 1:
            if cv_mode == 'kfc' : cv_mode = 'kf'
            if cv_mode == 'looc': cv_mode = 'loo'
        if cv_mode == 'loo': k = c.size
        
        # get CV matrix, if custom CV
        if cv_mode == 'custom':
            CV = cv_mat
            k  = CV.shape[1]
        # get class indices, otherwise
        else:
            n  = c.size
            C  = int(np.max(c))
            ic = [None] * C
            nc = np.zeros(C)
            for j in range(C):
                ic[j] = np.nonzero(c==j+1)
                ic[j] = np.array(ic[j][0], dtype=int)
                nc[j] = c[ic[j]].size
            CV = np.zeros((n,k))
        
        # k-folds and leave-one-out cross-validation
        if cv_mode == 'kf' or cv_mode == 'loo':
            nf = np.ceil(n/k)
            ia = np.arange(0,n)
            for g in range(k):
                i2 = np.arange(g*nf, np.min([(g+1)*nf, n]), dtype=int)
                i1 = [i for i in ia if i not in i2]
                CV[i1,g] = 1
                CV[i2,g] = 2
        
        # cross-validation points per class
        if cv_mode == 'kfc' or cv_mode == 'looc':
            nf = np.ceil(nc/k)
            ia = np.arange(0,n)
            for g in range(k):
                i2 = np.empty(1, dtype=int)
                for j in range(C):
                    ij = np.arange(g*nf[j], np.min([(g+1)*nf[j], nc[j]]), dtype=int)
                    i2 = np.append(i2, ic[j][ij])
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
        MBCR.predict(prior)
            prior - a dictionary specifying the prior distribution
            o x   - a  1 x L vector, the support of the distribution
            o p   - a  1 x L vector, the probability mass or density
             (L   - number of classes or number of target variable levels)
            
            MBCR  - an object for multivariate Bayesian classification or regression
            o PP  - an n x L matrix giving the posterior probabilities
            o xp  - an n x 1 vector giving maximum-a-posterior estimates
            o xt  - an n x 1 vector of true class indicies or target values
            
        This method predicts labels using multivariate Bayesian inversion for
        classification (MBC) or regression (MBR) using the data supplied via
        MBI.cvMBI and obeying the cross-validation structure specified via
        MBI.cvMBI.crossval.
        
        It stores a posterior probability matrix PP yielding the probability
        or probability density for each possible label value, given the
        observations for each data point.
        
        If prior is not supplied, it will be created as a uniform distribution
        over either class labels or target values (see MBI.model.test).
        """
        
        # set CV if required
        if not hasattr(self, 'CV'):
            self.crossval()
        k = self.CV.shape[1]
        
        # set prior if required
        if prior is None:
            if self.is_MBC:
                prior = uniprior('disc', int(np.max(self.x)))
            else:
                prior = uniprior('cont', 100, np.min(self.x), np.max(self.x))
        self.prior = prior
        
        # cross-validated analysis
        L  = prior['x'].size                # classes/levels
        xt = np.zeros(self.n)               # "true" classes
        xp = np.zeros(self.n)               # predicted classes
        PP = np.zeros((self.n,L))           # posterior probabilities
        for g in range(k):
            
            # get training and test set
            i1 = np.nonzero(self.CV[:,g]==1)
            i2 = np.nonzero(self.CV[:,g]==2)
            i1 = np.array(i1[0], dtype=int)
            i2 = np.array(i2[0], dtype=int)
            Y1 = self.Y[i1,:]
            Y2 = self.Y[i2,:]
            x1 = self.x[i1]
            x2 = self.x[i2]
            if self.X.shape[1] > 0:
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
            for i in range(i2.size):
                xt[i2[i]] = x2[i]
                xp[i2[i]] = prior['x'][PP[i2[i],:]==np.max(PP[i2[i],:])][0]
        
        # store prediction results
        self.xt = xt
        self.xp = xp
        self.PP = PP
        
    # function: performance evaluation
    #-------------------------------------------------------------------------#
    def evaluate(self, meas=None):
        """
        Evaluate Performance of Multivariate Bayesian Inversion
        perf = MBCR.evaluate(meas)
            meas - a string indicating the performance measure to use,
                   if classification (MBC)
                   o 'CA'  - classification accuracy (default)
                   o 'CAs' - class accuracies
                   o 'BA'  - balanced accuracy
                   or if regression (MBR)
                   o 'r'   - correlation coefficient (default)
                   o 'MAE' - mean absolute error
            
            perf - a scalar, the predictive performance of the model
            
        Each performance measure is calculated by comparing true labels with
        predicted labels (see MBI.cvMBI.predict) according to the measure.
        """
        
        # set measure if required
        if meas is None:
            if self.is_MBC: meas = 'CA'
            else:           meas = 'r'
        
        # calculate performance
        if meas == 'CA':                    # classification accuracy
            perf = np.mean(self.xp==self.xt)
        if meas == 'CAs':                   # class accuracies
            C    = int(np.max(self.xt))
            perf = np.zeros(C)
            for j in range(C):
                perf[j] = np.mean(self.xp[self.xt==(j+1)]==(j+1))
        if meas == 'BA':                    # balanced accuracy
            perf = np.mean(self.evaluate('CAs'))
        if meas == 'r':                     # correlation coefficient
            perf = np.corrcoef(self.xp, self.xt)[0,1]
        if meas == 'MAE':                   # mean absolute error
            perf = np.mean(np.absolute(self.xp-self.xt))
        
        # return performance
        return perf
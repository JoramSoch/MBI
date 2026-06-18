# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Simulation 4: comparison of methods (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 2022-02-20, 22:38: first version in MATLAB
- 2025-03-05, 11:00: final version in Python
- 2026-06-17, 15:46: rewrote MBR and SVR,
                     implemented RGA measure
- 2026-06-18, 13:34: added LinReg, RFC, NNC
- 2026-06-18, 20:15: added GNB regression
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
import scipy.linalg as sp_linalg
import scipy.stats as sp_stats
import sklearn as skl
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: specify ground truth ##############################################

# set ground truth
np.random.seed(1)
n  = 200                        # number of data points
k  = 10                         # number of CV folds
v  = 10                         # number of features
mu = 0                          # beta mean
sb = 1                          # beta variance
s2 = 1                          # noise variance
tau= 0.25                       # time constant and temporal covariance
V  = sp_linalg.toeplitz(np.power(tau, np.linspace(0,n-1,n)))
ny = 0.5                        # space constant and spatial covariance
Si = sp_linalg.toeplitz(np.power( ny, np.linspace(0,v-1,v)))

# generate targets
xm = 1                          # -1 < x < +1, x ~ U(-1,+1)
x  = np.random.uniform(-xm, +xm, size=(n,1))
X  = np.c_[x, np.ones(x.shape)] # design matrix
x  = np.squeeze(x)

# histogram of targets
lim = 1.5
dx  = 0.1
nx  = int((2*lim)/dx)+1
xb  = np.linspace(-lim, +lim, nx)
nb  = np.histogram(x, bins=xb)[0]


### Step 2: generate data #####################################################

# generate data
B = np.random.normal(mu, np.sqrt(sb), size=(X.shape[1],v))
E = MBI.matnrnd(np.zeros((n,v)), s2*V, Si, 1)
Y = X @ B + E


### Step 3: analyze data ######################################################

# specify analysis parameters
meth    = ['GNB', 'MBR', 'LinReg', 'SVR', 'RFR', 'NNR']
prior   = {}
prior['x'] = np.arange(-1, +1+0.01, 0.01)   # MBR prior distribution
prior['p'] = (1/(np.max(prior['x'])-np.min(prior['x'])))*np.ones(prior['x'].size)
Dgnb    = 'mvn'                             # GNB distribution name
Mlinreg = 'WLS'                             # LinReg estimation method
Ksvm    = 'linear'                          # SVM kernel type
Csvm    =  1                                # SVM cost parameter
Nrf     = 100                               # RF number of trees
Lnn     = (10, 10)                          # NN hidden layer sizes
Ann     = 'identity'                        # NN activation function

# specify cross-validation
MBR = MBI.cvMBI(Y, x, V=V, mb_type='MBR')
MBR.crossval(k=k, cv_mode='kf')             # k-fold cross-validation
CV  = MBR.CV

# preallocate results
M   = len(meth)
xp  = np.zeros((n,M))
r   = np.zeros(M)
MAE = np.zeros(M)
RGA = np.zeros(M)

# perform classification
for g in range(k):

    # get training and test set
    i1 = np.array(np.nonzero(CV[:,g]==1)[0], dtype=int)
    i2 = np.array(np.nonzero(CV[:,g]==2)[0], dtype=int)
    Y1 = Y[i1,:]
    Y2 = Y[i2,:]
    x1 = x[i1]
    x2 = x[i2]
    X1 = X[i1,:]
    X2 = X[i2,:]
    V1 = V[i1,:][:,i1]
    V2 = V[i2,:][:,i2]
    
    # evaluate all methods
    for h in range(M):
        
        # Method: multivariate Bayesian regression (MBI)
        # https://github.com/JoramSoch/MBI/blob/main/Python/MBI.py
        if meth[h] == 'MBR':
            mba1 = MBI.model(Y1, x1, V=V1, mb_type='MBR').train()
            pp2  = MBI.model(Y2, x2, V=V2, mb_type='MBR').test(mba1, prior)
            ip2  = np.argmax(pp2, axis=1)
            xp2  = prior['x'][ip2]
        
        # Method: Gaussian naive Bayes (custom)
        # https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model
        if meth[h] == 'GNB':
            B_est1  = np.linalg.inv(X1.T @ X1) @ X1.T @ Y1
            s2_est1 = (1/n) * np.sum(np.power(Y1 - X1 @ B_est1, 2), axis=0)
            Si_est1 = np.diag(s2_est1)
            log_PP  = np.zeros((i2.size, prior['x'].size))
            pp2     = np.zeros((i2.size, prior['x'].size))
            for i in range(i2.size):
                y2i = np.array([ Y2[i,:] ])
                for j in range(prior['x'].size):
                    x2ij        = np.array([ [prior['x'][j], 1] ])
                    ptd         = sp_stats.multivariate_normal(
                                      (x2ij @ B_est1)[0,:], Si_est1)
                    log_PP[i,j] = np.log(ptd.pdf(y2i))
                pp2[i,:] = np.exp(log_PP[i,:] - np.mean(log_PP[i,:]))
                pp2[i,:] = pp2[i,:] / np.trapz(pp2[i,:], prior['x'])
            ip2     = np.argmax(pp2, axis=1)
            xp2     = prior['x'][ip2]
        
        # Method: linear regression (Python)
        # https://github.com/JoramSoch/MBI/blob/main/Python/cvBMS.py
        if meth[h] == 'LinReg':
            P1  = np.linalg.inv(V1)
            if Mlinreg == 'OLS':
                b_est1 = np.linalg.inv(Y1.T @ Y1) @ Y1.T @ x1
            elif Mlinreg == 'WLS':
                b_est1 = np.linalg.inv(Y1.T @ P1 @ Y1) @ Y1.T @ P1 @ x1
            else:
                b_est1 = np.zeros((v,1))
            xp2 = Y2 @ b_est1

        # Method: support vector regression (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
        if meth[h] == 'SVR':
            svm1 = skl.svm.SVR(kernel=Ksvm, C=Csvm).fit(Y1, x1)
            xp2  = svm1.predict(Y2)
        
        # Method: random forrest regression (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
        if meth[h] == 'RFR':
            rf1 = skl.ensemble.RandomForestRegressor(
                n_estimators=Nrf).fit(Y1, x1)
            xp2 = rf1.predict(Y2)
        
        # Method: neural network regression (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
        if meth[h] == 'NNR':
            nn1 = skl.neural_network.MLPRegressor(
                hidden_layer_sizes=Lnn, activation=Ann).fit(Y1, x1)
            xp2 = nn1.predict(Y2)
        
        # store test set predictions
        xp[i2,h]   = xp2
        
    # delete analysis variables
    del mba1, pp2, ip2, B_est1, s2_est1, Si_est1, log_PP, P1, b_est1, svm1, rf1, nn1

# calculate performance
for h in range(M):
    r[h]   = np.corrcoef(xp[:,h], x)[0,1]
    MAE[h] = np.mean(np.absolute(xp[:,h]-x))
    RGA[h] = MBI.calc_RGA(x, xp[:,h], 'RGA')


### Step 3: visualize results #################################################

# open figure
fig   = plt.figure(figsize=(8,9))
ax    = fig.add_subplot(111)
x_off = 2
y_off = 0.98
cols  = [[  0,  32,  96], [  0,   0, 255],
         [192,   0,   0], [255,   0,   0], [255, 192,   0], [255, 255,   0]]

# plot performance
for h in range(M):
    col = tuple([rgb/255 for rgb in cols[h]])
    ax.bar(h+1, RGA[h], width= 0.7, color=col, edgecolor='k', label=meth[h])
ax.plot([(1-1), (M+2)], [1/2, 1/2],     ':k', linewidth=2, label='chance')
ax.plot([x_off+1/2, x_off+1/2], [0, 1], '-k', linewidth=1)
ax.axis([(1-1), (M+2), 0, 1])
ax.set_xticks([])
ax.legend(loc='lower right', fontsize=12)
ax.set_xlabel('classification approach', fontsize=20)
ax.set_ylabel('rank graduation accuracy', fontsize=20)
ax.set_title('Simulation 4', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.text(x_off+1/2, y_off, 'generative methods   ',
        fontsize=14, ha='right', va='center')
ax.text(x_off+1/2, y_off, '   discriminative methods',
        fontsize=14, ha='left', va='center')

# enable tight layout
fig.tight_layout()
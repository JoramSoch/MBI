# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Analysis 4: comparison of methods (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 2022-02-21, 20:29: first version in MATLAB
- 2025-03-05, 18:45: final version in Python
- 2026-06-22, 15:08: rewrote MBR and SVR,
                     implemented RGA measure
- 2026-06-22, 15:30: added GNB, LinReg, RFR, NNR                     
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
import scipy as sp
import sklearn as skl
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: load data #########################################################

# load data
train = sp.io.loadmat('../Analysis_4/PAC_specify.mat')
test  = sp.io.loadmat('../Analysis_4/PAC_specify_test_age.mat')

# assemble data (MBR)
n1  = train['sID1'].size                    # number of data points
n2  = train['sID2'].size
V1  = np.eye(n1)                            # observation covariances
V2  = np.eye(n2)
YA1 = np.c_[train['GM1'], train['WM1']]     # data matrices
YA2 = np.c_[train['GM2'], train['WM2']]
x1  = np.squeeze(train['y1'])               # label vectors
x2  = np.squeeze(test['y2'])
X1  = np.c_[x1, np.ones((n1,1))]            # design matrices
X2  = np.c_[x2, np.ones((n2,1))]
XA1 = train['c1'][:,1:]                     # covariate matrices
XA2 = train['c2'][:,1:]

# assemble data (SVR)                       # feature matrices
YB1 = np.c_[train['GM1'], train['WM1'], train['c1']]
YB2 = np.c_[train['GM2'], train['WM2'], train['c2']]
del train, test

# prepare histograms
x_min = 0
x_max = 100
dx    = 2.5
nx    = int((x_max-x_min)/dx)+1
xb    = np.linspace(x_min, x_max, nx)


### Step 2: analyze data ######################################################

# specify analysis parameters
meth    = ['GNB', 'MBR', 'LinReg', 'SVR', 'RFR', 'NNR']
prior   = {}
prior['x'] = np.arange(0, 100+1, 1)         # MBR prior distribution
prior['p'] = (1/(np.max(prior['x'])-np.min(prior['x'])))*np.ones(prior['x'].size)
Dgnb    = 'mvn'                             # GNB distribution name
Mlinreg = 'WLS'                             # LinReg estimation method
Ksvm    = 'linear'                          # SVM kernel type
Csvm    =  1                                # SVM cost parameter
Nrf     = 100                               # RF number of trees
Lnn     = (YB1.shape[1],)                   # NN hidden layer sizes
Ann     = 'identity'                        # NN activation function

# preallocate results
M   = len(meth)
xp  = np.zeros((n2,M))
r   = np.zeros(M)
MAE = np.zeros(M)
RGA = np.zeros(M)

# evaluate all methods
for h in range(M):
    
    # Method: multivariate Bayesian regression (MBI)
    # https://github.com/JoramSoch/MBI/blob/main/Python/MBI.py
    if meth[h] == 'MBR':
        mba1 = MBI.model(YA1, x1, V=V1, mb_type='MBR').train()
        pp2  = MBI.model(YA2, x2, V=V2, mb_type='MBR').test(mba1, prior)
        ip2  = np.argmax(pp2, axis=1)
        xp2  = prior['x'][ip2]
    
    # Method: Gaussian naive Bayes (custom)
    # https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model
    if meth[h] == 'GNB':
        B_est1  = np.linalg.inv(X1.T @ X1) @ X1.T @ YA1
        s2_est1 = (1/n1) * np.sum(np.power(YA1 - X1 @ B_est1, 2), axis=0)
        Si_est1 = np.diag(s2_est1)
        log_PP  = np.zeros((n2, prior['x'].size))
        pp2     = np.zeros((n2, prior['x'].size))
        for i in range(n2):
            y2i = np.array([ YA2[i,:] ])
            for j in range(prior['x'].size):
                x2ij        = np.array([ [prior['x'][j], 1] ])
                ptd         = sp.stats.multivariate_normal(
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
            b_est1 = np.linalg.inv(YB1.T @ YB1) @ YB1.T @ x1
        elif Mlinreg == 'WLS':
            b_est1 = np.linalg.inv(YB1.T @ P1 @ YB1) @ YB1.T @ P1 @ x1
        else:
            b_est1 = np.zeros((YB1.shape[1],1))
        xp2 = YB2 @ b_est1
    
    # Method: support vector regression (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
    if meth[h] == 'SVR':
        svm1 = skl.svm.SVR(kernel=Ksvm, C=Csvm).fit(YB1, x1)
        xp2  = svm1.predict(YB2)
    
    # Method: random forrest regression (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
    if meth[h] == 'RFR':
        rf1 = skl.ensemble.RandomForestRegressor(
            n_estimators=Nrf).fit(YB1, x1)
        xp2 = rf1.predict(YB2)
    
    # Method: neural network regression (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
    if meth[h] == 'NNR':
        nn1 = skl.neural_network.MLPRegressor(
            hidden_layer_sizes=Lnn, activation=Ann).fit(YB1, x1)
        xp2 = nn1.predict(YB2)
    
    # store test set predictions
    xp[:,h] = xp2
    r[h]    = np.corrcoef(xp[:,h], x2)[0,1]
    MAE[h]  = np.mean(np.absolute(xp[:,h]-x2))
    RGA[h]  = MBI.calc_RGA(x2, xp[:,h], 'RGA')

# delete analysis variables
del mba1, pp2, ip2, B_est1, s2_est1, Si_est1, log_PP, P1, b_est1, svm1, rf1, nn1


### Step 3: visualize results #################################################

# open figure
fig   = plt.figure(figsize=(8,9))
ax    = fig.add_subplot(111)
x_off = 2
y_off = 0.975
cols  = [[  0,  32,  96], [  0,   0, 255],
         [192,   0,   0], [255,   0,   0], [255, 192,   0], [255, 255,   0]]

# plot performance
for h in range(M):
    col = tuple([rgb/255 for rgb in cols[h]])
    ax.bar(h+1, RGA[h], width=0.7, color=col, edgecolor='k', label=meth[h])
ax.plot([(1-1), (M+2)], [1/2, 1/2],     ':k', linewidth=2, label='chance')
ax.plot([x_off+1/2, x_off+1/2], [0, 1], '-k', linewidth=1)
ax.axis([(1-1), (M+2), 0, 1])
ax.set_xticks([])
ax.legend(loc='lower right', fontsize=12)
ax.set_xlabel('regression approach', fontsize=20)
ax.set_ylabel('rank graduation accuracy', fontsize=20)
ax.set_title('Analysis 4', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.text(x_off+1/2, y_off, 'generative methods   ',
        fontsize=14, ha='right', va='center')
ax.text(x_off+1/2, y_off, '   discriminative methods',
        fontsize=14, ha='left', va='center')

# enable tight layout
fig.tight_layout()
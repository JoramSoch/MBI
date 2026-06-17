# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Simulation 3: comparison of methods (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 2022-02-20, 12:23: first version in MATLAB
- 2025-03-05, 10:58: final version in Python
- 2026-06-16, 19:49: rewrote MBC and SVC,
                     implemented RGA measure
- 2026-06-17, 12:38: added GNB, LDA, LogReg
- 2026-06-17, 14:27: added RFC, NNC
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
import sklearn as skl
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: specify ground truth ##############################################

# set ground truth
np.random.seed(2)
mu = 1                                      # class means
Si = np.array([[1,0.5],[0.5,1]])            # feature covariance
s2 = 4                                      # noise variance
n  = 300                                    # number of data points
k  = 10                                     # number of CV folds
v  = 2                                      # number of features
C  = 3                                      # number of classes

# generate classes
x  = np.kron(np.arange(C).reshape((C,1))+1, np.ones((int(n/C),1)))
x  = np.random.permutation(x[:,0])          # randomized labels
X  = np.zeros((n,C))                        # design matrix
V  = np.eye(n)                              # observation covariance
for i in range(n):
    X[i,int(x[i]-1)] = 1


### Step 2: generate data #####################################################

# generate data
B = np.array([[  -mu,   +mu],
              [+2*mu, +2*mu],
              [  +mu,   -mu]])
E = MBI.matnrnd(np.zeros((n,v)), s2*V, Si, 1)
Y = X @ B + E


### Step 3: analyze data ######################################################

# specify analysis parameters
meth    = ['GNB', 'MBC', 'LDA', 'LogReg', 'SVC', 'RFC', 'NNC']
prior   = {'x': np.arange(C)+1,             # MBC class indices
           'p': 1/C*np.ones(C)}             # MBC prior probabilities
Dgnb    = 'normal'                          # GNB distribution name
Dlda    = 'linear'                          # LDA discriminant type
Llogreg =  1                                # LogReg cost parameter
Ksvm    = 'linear'                          # SVM kernel type
Csvm    =  1                                # SVM cost parameter
Nrf     = 100                               # RF number of trees
Lnn     = (10, 10)                          # NN hidden layer sizes
Ann     = 'logistic'                        # NN activation function

# specify cross-validation
MBC = MBI.cvMBI(Y, x, V=V, mb_type='MBC')
MBC.crossval(k=k, cv_mode='kfc')            # k-fold CV on points per class
CV  = MBC.CV

# preallocate results
M   = len(meth)
xp  = np.zeros((n,M))
xs  = np.zeros((n,C,M))
CA  = np.zeros(M)
RGA = np.zeros(M)

# perform classification
for g in range(k):

    # get training and test set
    i1 = np.array(np.nonzero(MBC.CV[:,g]==1)[0], dtype=int)
    i2 = np.array(np.nonzero(MBC.CV[:,g]==2)[0], dtype=int)
    Y1 = Y[i1,:]
    Y2 = Y[i2,:]
    x1 = x[i1]
    x2 = x[i2]
    V1 = V[i1,:][:,i1]
    V2 = V[i2,:][:,i2]
    
    # evaluate all methods
    for h in range(M):
        
        # prepare test set predictions
        xp2  = np.zeros(i2.size)
        xs2  = 1/C*np.ones((i2.size,C))
        
        # Method: multivariate Bayesian classification (MBI)
        # https://github.com/JoramSoch/MBI/blob/main/Python/MBI.py
        if meth[h] == 'MBC':
            mba1 = MBI.model(Y1, x1, V=V1, mb_type='MBC').train()
            pp2  = MBI.model(Y2, x2, V=V2, mb_type='MBC').test(mba1, prior)
            xp2  = np.argmax(pp2, axis=1) + 1
            xs2  = pp2
        
        # Method: Gaussian naive Bayes (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
        if meth[h] == 'GNB':
            gnb1 = skl.naive_bayes.GaussianNB().fit(Y1, x1)
            xp2  = gnb1.predict(Y2)
            xs2  = gnb1.predict_proba(Y2)

        # Method: linear discriminant analysis (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
        if meth[h] == 'LDA':
            lda1 = skl.discriminant_analysis.LinearDiscriminantAnalysis().fit(Y1, x1)
            xp2  = lda1.predict(Y2)
            xs2  = lda1.predict_proba(Y2)

        # Method: logistic regression (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
        if meth[h] == 'LogReg':
            logreg1 = skl.linear_model.LogisticRegression(C=Llogreg).fit(Y1, x1)
            xp2     = logreg1.predict(Y2)
            xs2     = logreg1.decision_function(Y2)

        # Method: support vector classification (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
        if meth[h] == 'SVC':
            svm1 = skl.svm.SVC(kernel=Ksvm, C=Csvm).fit(Y1, x1)
            xp2  = svm1.predict(Y2)
            xs2  = svm1.decision_function(Y2)
        
        # Method: random forrest classification (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
        if meth[h] == 'RFC':
            rf1 = skl.ensemble.RandomForestClassifier(
                n_estimators=Nrf).fit(Y1, x1)
            xp2 = rf1.predict(Y2)
            xs2 = rf1.predict_proba(Y2)
        
        # Method: neural network classification (scikit-learn)
        # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
        if meth[h] == 'NNC':
            nn1 = skl.neural_network.MLPClassifier(
                hidden_layer_sizes=Lnn, activation=Ann).fit(Y1, x1)
            xp2 = nn1.predict(Y2)
            xs2 = nn1.predict_proba(Y2)
        
        # store test set predictions
        xp[i2,h]   = xp2
        xs[i2,:,h] = xs2
        
    # delete analysis variables  
    del mba1, pp2, gnb1, lda1, svm1

# calculate performance
for h in range(M):
    CA[h]  = np.mean(xp[:,h]==x)
    RGA[h] = MBI.calc_RGA(x, xs[:,:,h], 'macro')


### Step 4: visualize results #################################################

# open figure
fig   = plt.figure(figsize=(8,9))
ax    = fig.add_subplot(111)
x_off = 3
y_off = 0.9
cols  = [[  0,  32,  96], [  0,   0, 255], [  0, 176, 240],
         [192,   0,   0], [255,   0,   0], [255, 192,   0], [255, 255,   0]]

# plot performance
for h in range(M):
    col = tuple([rgb/255 for rgb in cols[h]])
    ax.bar(h+1, RGA[h], width= 0.7, color=col, edgecolor='k', label=meth[h])
ax.plot([(1-1), (M+1)], [1/2, 1/2],     ':k', linewidth=2, label='chance')
ax.plot([x_off+1/2, x_off+1/2], [0, 1], '-k', linewidth=1)
ax.axis([(1-1), (M+1), 0, 1])
ax.set_xticks([])
ax.legend(loc='lower right', fontsize=12)
ax.set_xlabel('classification approach', fontsize=20)
ax.set_ylabel('rank graduation accuracy', fontsize=20)
ax.set_title('Simulation 3', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.text(x_off+1/2, y_off, 'generative   \nmethods   ',
        fontsize=20, ha='right', va='center')
ax.text(x_off+1/2, y_off, '   discriminative\n   methods',
        fontsize=20, ha='left', va='center')

# enable tight layout
fig.tight_layout()
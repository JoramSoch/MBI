# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Analysis 2: comparison of methods (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 2025-01-03, 16:57: first version in MATLAB
- 2025-03-05, 18:44: final version in Python
- 2026-06-19, 12:03: rewrote MBC and SVC,
                     implemented RGA measure
- 2026-06-19, 14:45: added GNB, LDA, LogReg, RFC, NNC
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


### Step 1: load data #########################################################

# load extracted data
data = np.load('../Analysis_2/MNIST_data.npz')
Y1   = data['Y1']
Y2   = data['Y2']
x1   = data['x1']
x2   = data['x2']
del data

# get data dimensions
n1   = Y1.shape[0]              # number of training data points
n2   = Y2.shape[0]              # number of test data points
v    = Y2.shape[1]              # number of features
C    = np.max(x2)               # number of classes


### Step 2: analyze data ######################################################

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
Lnn     = (v,)                              # NN hidden layer sizes
Ann     = 'logistic'                        # NN activation function

# restrict to data subset
N1  = 60000                                 # training data points to use
N2  = 10000                                 # test data points to use

# preallocate results
M   = len(meth)
xp  = np.zeros((N2,M))
xs  = np.zeros((N2,C,M))
CA  = np.zeros(M)
RGA = np.zeros(M)

# display parameters
print('\n-> Analysis 2:')
print('   - N1 = {} training data points'.format(N1))
print('   - N2 = {} test data points'.format(N2))

# evaluate all methods
for h in range(M):
    
    # display method
    print('   - {}: '.format(meth[h]), end='')
    
    # Method: multivariate Bayesian classification (MBI)
    # https://github.com/JoramSoch/MBI/blob/main/Python/MBI.py
    if meth[h] == 'MBC':
        print('training ... ', end='')
        mba1 = MBI.model(Y1[:N1,:], x1[:N1], mb_type='MBC').train()
        print('testing ... ', end='')
        pp2  = MBI.model(Y2[:N2,:], x2[:N2], mb_type='MBC').test(mba1, prior)
        xp2  = np.argmax(pp2, axis=1) + 1
        xs2  = pp2
        print('done.')
    
    # # Method: Gaussian naive Bayes (scikit-learn)
    # # https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    if meth[h] == 'GNB':
        # remove features
        print('remove at-least-one-class zero-variance features ', end='')
        Y1h = Y1[:N1,:]
        x1h = x1[:N1]
        Y1v = np.zeros((C,v))
        for k in range(C):
            Y1v[k,:] = np.var(Y1h[x1h==k+1], axis=0)
        jh  = np.all(Y1v>0, axis=0)
        print('({})'.format(v-np.sum(jh)))
        print('          ', end='')
        # apply classifier
        print('training ... ', end='')
        Y1h  = Y1h[:,jh]
        gnb1 = skl.naive_bayes.GaussianNB().fit(Y1h, x1h)
        print('testing ... ', end='')
        xp2  = gnb1.predict(Y2[:N2,jh])
        xs2  = gnb1.predict_proba(Y2[:N2,jh])
        print('done.')
    
    # Method: linear discriminant analysis (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.LinearDiscriminantAnalysis.html
    if meth[h] == 'LDA':
        print('training ... ', end='')
        lda1 = skl.discriminant_analysis.LinearDiscriminantAnalysis().fit(Y1[:N1,:], x1[:N1])
        print('testing ... ', end='')
        xp2  = lda1.predict(Y2[:N2,:])
        xs2  = lda1.predict_proba(Y2[:N2,:])
        print('done.')
    
    # Method: logistic regression (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    if meth[h] == 'LogReg':
        print('training ... ')
        logreg1 = skl.linear_model.LogisticRegression(C=Llogreg).fit(Y1[:N1,:], x1[:N1])
        print('testing ... ', end='')
        xp2     = logreg1.predict(Y2[:N2,:])
        xs2     = logreg1.decision_function(Y2[:N2,:])
        print('done.')
    
    # Method: support vector classification (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
    if meth[h] == 'SVC':
        print('training ... ', end='')
        svm1 = skl.svm.SVC(kernel=Ksvm, C=Csvm).fit(Y1[:N1,:], x1[:N1])
        print('testing ... ', end='')
        xp2  = svm1.predict(Y2[:N2,:])
        xs2  = svm1.decision_function(Y2[:N2,:])
        print('done.')
    
    # Method: random forrest classification (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    if meth[h] == 'RFC':
        print('training ... ', end='')
        rf1 = skl.ensemble.RandomForestClassifier(
            n_estimators=Nrf).fit(Y1[:N1,:], x1[:N1])
        print('testing ... ', end='')
        xp2 = rf1.predict(Y2[:N2,:])
        xs2 = rf1.predict_proba(Y2[:N2,:])
        print('done.')
    
    # Method: neural network classification (scikit-learn)
    # https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
    if meth[h] == 'NNC':
        print('training ... ', end='')
        nn1 = skl.neural_network.MLPClassifier(
            hidden_layer_sizes=Lnn, activation=Ann).fit(Y1[:N1,:], x1[:N1])
        print('testing ... ', end='')
        xp2 = nn1.predict(Y2[:N2,:])
        xs2 = nn1.predict_proba(Y2[:N2,:])
        print('done.')
    
    # calculate performance
    xp[:,h]   = xp2
    xs[:,:,h] = xs2
    CA[h]     = np.mean(xp[:,h]==x2[:N2])
    RGA[h]    = MBI.calc_RGA(x2[:N2], xs[:,:,h], 'macro')
    
# delete analysis variables  
del mba1, pp2, Y1h, x1h, Y1v, gnb1, lda1, logreg1, svm1, rf1, nn1


### Step 3: visualize results #################################################

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
    ax.bar(h+1, RGA[h], width=0.7, color=col, edgecolor='k', label=meth[h])
ax.plot([(1-1), (M+1)], [1/2, 1/2],     ':k', linewidth=2, label='chance')
ax.plot([x_off+1/2, x_off+1/2], [0, 1], '-k', linewidth=1)
ax.axis([(1-1), (M+1), 0, 1])
ax.set_xticks([])
ax.legend(loc='lower right', fontsize=12)
ax.set_xlabel('classification approach', fontsize=20)
ax.set_ylabel('rank graduation accuracy', fontsize=20)
ax.set_title('Analysis 2', fontsize=24, fontweight='bold')
ax.tick_params(axis='both', labelsize=12)
ax.text(x_off+1/2, y_off, 'generative   \nmethods   ',
        fontsize=20, ha='right', va='center')
ax.text(x_off+1/2, y_off, '   discriminative\n   methods',
        fontsize=20, ha='left', va='center')

# enable tight layout
fig.tight_layout()
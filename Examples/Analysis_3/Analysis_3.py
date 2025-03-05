# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Analysis 3: birth weight data (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 21/02/2022, 19:10: first version in MATLAB
- 27/02/2025, 11:20: ported code to Python (Step 1 & 2)
- 27/02/2025, 12:16: ported code to Python (Step 3)
- 05/03/2025, 11:03: added to GitHub repository
- 05/03/2025, 15:23: recoded confusion matrix
- 05/03/2025, 18:40: unified tick font sizes
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
import pandas as pd
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: load data #########################################################

# load TSV file
filename = 'Birth_Weights.csv'
data = pd.read_csv(filename, sep=',')
hdr  = list(data)                           # column names

# extract data
Y = np.array(data[['Birth_weight','Weight']])# data matrix
X = np.zeros((Y.shape[0],5))                # design matrix
X[:,0] = 1*(data['Smoker']=='no') + 2*(data['Smoker']=='yes')
X[:,1] = 1*(data['Race']=='white') + 2*(data['Race']=='black') + 3*(data['Race']=='other')
X[:,2] = 1*(data['Hypertension']=='no') + 2*(data['Hypertension']=='yes')
X[:,3] = data['Age']
X[:,4] = data['Visits']
n = Y.shape[0]                              # number of data points
v = Y.shape[1]                              # number of features


### Step 2: analyze data ######################################################

# specify parameters
k = 10                                      # number of CV folds
V = np.eye(n)                               # observation covariance

# preallocate results
xt  = [X[:,0], X[:,1], X[:,2]]
MBC = [None for h in range(len(xt))]
SVC = [None for h in range(len(xt))]
BA  = np.zeros((2,len(xt)))
nC  = [int(np.max(x)) for x in xt]

# Analysis 1: classify smoker, (not) accounting for others
x1  = X[:,0]
X1  = np.c_[1*(X[:,1]==1)-1*(X[:,1]==2), 1*(X[:,1]==2)-1*(X[:,1]==3),
            1*(X[:,2]==1)-1*(X[:,2]==2), X[:,3:5]]
X1r = np.c_[X1, np.ones((n,1))]
Y1r = (np.eye(n) - X1r @ np.linalg.inv(X1r.T @ X1r) @ X1r.T) @ Y
MBC[0]  = MBI.cvMBI(Y, x1, X=X1, V=V, mb_type='MBC')
MBC[0].crossval(k=k, cv_mode='kfc')
MBC[0].predict()
BA[0,0] = MBC[0].evaluate('BA')

# Analysis 2: classify ethnicity, (not) accounting for others
x2  = X[:,1]
X2  = np.c_[1*(X[:,0]==1)-1*(X[:,0]==2), 1*(X[:,2]==1)-1*(X[:,2]==2), X[:,3:5]]
X2r = np.c_[X2, np.ones((n,1))]
Y2r = (np.eye(n) - X2r @ np.linalg.inv(X2r.T @ X2r) @ X2r.T) @ Y
MBC[1]  = MBI.cvMBI(Y, x2, X=X2, V=V, mb_type='MBC')
MBC[1].crossval(k=k, cv_mode='kfc')
MBC[1].predict()
BA[0,1] = MBC[1].evaluate('BA')

# Analysis 3: classify hypertension, (not) accounting for others
x3  = X[:,2]
X3  = np.c_[1*(X[:,0]==1)-1*(X[:,0]==2), 
            1*(X[:,1]==1)-1*(X[:,1]==2), 1*(X[:,1]==2)-1*(X[:,1]==3), X[:,3:5]]
X3r = np.c_[X3, np.ones((n,1))]
Y3r = (np.eye(n) - X3r @ np.linalg.inv(X3r.T @ X3r) @ X3r.T) @ Y
MBC[2]  = MBI.cvMBI(Y, x3, X=X3, V=V, mb_type='MBC')
MBC[2].crossval(k=k, cv_mode='kfc')
MBC[2].predict()
BA[0,2] = MBC[2].evaluate('BA')

# Analyses 1-3: support vector classifications
Yr  = [Y1r, Y2r, Y3r]
xp  = [None for h in range(len(xt))]
for h in range(len(xt)):
    Yh  = Yr[h]
    xh  = xt[h]
    xph = np.zeros(xt[h].size)
    for g in range(k):
        # get training and test set
        i1  = np.array(np.nonzero(MBC[h].CV[:,g]==1)[0], dtype=int)
        i2  = np.array(np.nonzero(MBC[h].CV[:,g]==2)[0], dtype=int)
        Y1  = Yh[i1,:]
        Y2  = Yh[i2,:]
        x1  = xh[i1]
        # train and test using SVC
        SVC = svm.SVC(kernel='linear', C=1)
        SVC.fit(Y1, x1)
        xph[i2] = SVC.predict(Y2)
    xp[h]   = xph
    CAs     = np.array([np.mean(xph[xh==j+1]==j+1) for j in range(nC[h])])
    BA[1,h] = np.mean(CAs)
del Yh, xh, xph, SVC, CAs


### Step 3: visualize results #################################################

# create colormap
dx   = 0.1
dc   = 0.01
ns   = int(1/dc)
cmap = np.r_[np.c_[np.linspace(0, 1-dc, ns), np.linspace(0, 1-dc, ns), np.linspace(0, 1-dc, ns)],
             np.array([[1, 1, 1]]),
             np.c_[np.linspace(1-dc, 0, ns), np.ones((ns,1)), np.linspace(1-dc, 0, ns)]]
cmo2 = mpl.colors.ListedColormap(cmap)
cmap = np.r_[np.c_[np.linspace(0, 1-2*dc, int(ns/2)), np.linspace(0, 1-2*dc, int(ns/2)), np.linspace(0, 1-2*dc, int(ns/2))],
             np.array([[1, 1, 1]]),
             np.c_[np.linspace(1-dc, 0, ns), np.ones((ns,1)), np.linspace(1-dc, 0, ns)]]
cmo3 = mpl.colors.ListedColormap(cmap)

# specify labels
cols = 'brgcm'
comp = ['smok.','ethn.','tension']
labs = [['non-smoker','smoker'],
        ['white','black','other'],
        ['normal tension','hypertension']]

# open figure
fig = plt.figure(figsize=(16,10))
axs = fig.subplots(3, 5, width_ratios=[1,1,1,1.5,1.5])

# data set (classes)
y_lab = ['birth weight', 'mother\'s weight']
for k in range(len(labs)):
    for j in range(nC[k]):
        axs[k,0].plot(Y[xt[k]==j+1,1], Y[xt[k]==j+1,0], '.',
                      markersize=5, color=cols[j], label=labs[k][j])
    axs[k,0].set_xlim(np.min(Y[:,1])-(1/20)*np.ptp(Y[:,1]),
                      np.max(Y[:,1])+(1/20)*np.ptp(Y[:,1]))
    axs[k,0].set_ylim(np.min(Y[:,0])-(1/20)*np.ptp(Y[:,0]),
                      np.max(Y[:,0])+(1/20)*np.ptp(Y[:,0]))
    axs[k,0].legend(loc='upper right')
    axs[k,0].set_xlabel(y_lab[1], fontsize=16)
    axs[k,0].set_ylabel(y_lab[0], fontsize=16)
    axs[k,0].tick_params(axis='both', labelsize=10)
    if k == 0:
        axs[k,0].set_title('Data Set', fontsize=16, fontweight='bold')
axs[0,1].axis('off')

# data set (targets)
x_lab = ['mother\'s age', 'visits to the doctor']
for k1 in range(len(x_lab)):
    for k2 in range(len(y_lab)):
        xk = X[:,3+k1]
        yk = Y[:,0+k2]
        rk = np.corrcoef(xk, yk)[0,1]
        axs[1+k2,1+k1].plot(xk, yk, '.',
                            markersize=5, color=cols[3+k1])
        axs[1+k2,1+k1].set_xlim(np.min(xk)-(1/20)*np.ptp(xk),
                                np.max(xk)+(1/20)*np.ptp(xk))
        axs[1+k2,1+k1].set_ylim(np.min(yk)-(1/20)*np.ptp(yk),
                                np.max(yk)+(1/20)*np.ptp(yk))
        axs[1+k2,1+k1].set_xlabel(x_lab[k1], fontsize=16)
        axs[1+k2,1+k1].set_ylabel(y_lab[k2], fontsize=16)
        axs[1+k2,1+k1].tick_params(axis='both', labelsize=10)
        axs[1+k2,1+k1].text(np.max(xk), np.max(yk), 'r = {:.2f}'.format(rk),
                            fontsize=12, ha='right', va='top')

# confusion matrices
for h in range(len(xt)):
    for g in range(2):
        if g == 0:
            CM = MBC[h].evaluate('CM')
        if g == 1:
            CM = np.array([[np.mean(xp[h][xt[h]==j1]==j2)
                            for j1 in range(1,nC[h]+1)]
                            for j2 in range(1,nC[h]+1)])
        if nC[h] == 2:
            no = mpl.colors.Normalize(vmin=0, vmax=1)
            sm = mpl.cm.ScalarMappable(norm=no, cmap=cmo2)
            axs[h,3+g].matshow(CM, aspect='equal', cmap=cmo2, norm=no)
        if nC[h] == 3:
            no = mpl.colors.Normalize(vmin=0, vmax=1)
            sm = mpl.cm.ScalarMappable(norm=no, cmap=cmo3)
            axs[h,3+g].matshow(CM, aspect='equal', cmap=cmo3, norm=no)
        axs[h,3+g].xaxis.set_ticks_position('bottom')
        axs[h,3+g].set_xticks(range(nC[h]), labels=labs[h])
        axs[h,3+g].set_yticks(range(nC[h]), labels=[str(j) for j in range(1,nC[h]+1)])
        axs[h,3+g].set_xlabel('true class', fontsize=16)
        axs[h,3+g].set_ylabel('predicted class', fontsize=16)
        axs[h,3+g].tick_params(axis='both', labelsize=10)
        if h == 0:
            title = '{}'.format(['MBC with covariates','SVC with regression'][g])
            axs[h,3+g].set_title(title, fontsize=16, fontweight='bold')
        fig.colorbar(sm, ax=axs[h,3+g], orientation='vertical', label='')
        for c1 in range(nC[h]):
            for c2 in range(nC[h]):
                axs[h,3+g].text(c1, c2, '{:.2f}'.format(CM[c2,c1]),
                                fontsize=10, ha='center', va='center')

# classification accuracies
axs[0,2].bar(np.arange(3)-1.5*dx, BA[0,:],
             width=2*dx, align='center', color=(0,0,1), edgecolor='k', label='MBC')
axs[0,2].bar(np.arange(3)+1.5*dx, BA[1,:],
             width=2*dx, align='center', color=(1,0,0), edgecolor='k', label='SVC')
axs[0,2].plot([-1, 0.5, 0.5, 1.5, 1.5, 3], [1/2, 1/2, 1/3, 1/3, 1/2, 1/2], ':k',
              linewidth=2, label='chance')
axs[0,2].axis([(0-1), len(nC), 0, 1])
axs[0,2].set_xticks(range(3), labels=comp)
handles, labels = axs[0,2].get_legend_handles_labels(); o = [1,2,0]
axs[0,2].legend([handles[i] for i in o], [labels[i] for i in o],
                loc='upper left', fontsize=10)
axs[0,2].set_xlabel('classified variable', fontsize=16)
axs[0,2].set_ylabel('balanced accuracy', fontsize=16)
axs[0,2].set_title('Classification', fontsize=16, fontweight='bold')
axs[0,2].tick_params(axis='both', labelsize=10)

# enable tight layout
fig.tight_layout()
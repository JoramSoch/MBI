# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Analysis 1: Egyptian skull data (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 21/02/2022, 00:41: first version in MATLAB
- 25/02/2025, 16:14: ported code to Python (Step 1 & 2)
- 26/02/2025, 11:18: ported code to Python (Step 3)
- 05/03/2025, 11:01: added to GitHub repository
- 05/03/2025, 15:21: recoded confusion matrix
- 05/03/2025, 18:38: unified tick font sizes
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
filename = 'Egyptian_Skulls.tsv'
data = pd.read_csv(filename, sep='\t')
hdr  = list(data)               # column names

# extract data
xC = np.unique(data[hdr[-1]])   # class labels
x  = np.array(data[hdr[-1]])    # label vector
Y  = np.array(data[hdr[:-1]])   # data matrix
n  = Y.shape[0]                 # number of data points
v  = Y.shape[1]                 # number of features

# assign classes
C  = xC.size
for j in range(C):              # replace class labels
    x[x==xC[j]] = j+1           # by 1, 2, 3, ...


### Step 2: analyze data ######################################################

# specify parameters
k = 10                          # number of CV folds
V = np.eye(n)                   # observation covariance

# specify analyses
iC    = [None for h in range(3)]
it    = [None for h in range(3)]
xt    = [None for h in range(3)]
iC[0] = [1, 2, 3, 4, 5]
it[0] = [i for i in range(n) if x[i] in iC[0]]
xt[0] = x[it[0]]
iC[1] = [1, 3, 5]
it[1] = [i for i in range(n) if x[i] in iC[1]]
xt[1] = x[it[1]]; xt[1][xt[1]==3] = 2; xt[1][xt[1]==5] = 3;
iC[2] = [1, 5]
it[2] = [i for i in range(n) if x[i] in iC[2]]
xt[2] = x[it[2]]; xt[2][xt[2]==5] = 2;

# preallocate results
MBC = [None for h in range(len(iC))]
SVC = [None for h in range(len(iC))]
CA  = np.zeros((2,len(iC)))
nC  = [len(i) for i in iC]

# Analyses 1: MBC
for h in range(len(iC)):
    Vh      = np.eye(xt[h].size)
    MBC[h]  = MBI.cvMBI(Y[it[h],:], xt[h], V=Vh, mb_type='MBC')
    MBC[h].crossval(k=k, cv_mode='kfc')
    MBC[h].predict()
    CA[0,h] = MBC[h].evaluate('CA')
del Vh

# Analyses 2: SVC
xp  = [None for h in range(3)]
for h in range(len(iC)):
    Yh  = Y[it[h],:]
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
    CA[1,h] = np.mean(xph==xh)
del Yh, xh, xph, SVC


### Step 3: visualize results #################################################

# create colormap
dx   = 0.1
dc   = 0.01
ns   = int(1/dc)
cmap = np.r_[np.c_[np.linspace(0, 1-dc, ns), np.linspace(0, 1-dc, ns), np.linspace(0, 1-dc, ns)],
             np.array([[1, 1, 1]]),
             np.c_[np.linspace(1-dc, 0, ns), np.ones((ns,1)), np.linspace(1-dc, 0, ns)]]
cmo  = mpl.colors.ListedColormap(cmap)
cols = 'rgbcm'

# open figure
fig = plt.figure(figsize=(16,10))
axs = fig.subplots(3, 5, width_ratios=[1,1,1,1.5,1.5])

# data set
for k1 in range(0,v-1):
    for k2 in range(k1+1,v):
        for j in range(C):
            axs[k2-1,k1].plot(Y[x==j+1,k1], Y[x==j+1,k2], '.',
                              markersize=5, color=cols[j], label='{}'.format(xC[j]))
        axs[k2-1,k1].set_xlim(np.min(Y[:,k1])-(1/20)*np.ptp(Y[:,k1]),
                              np.max(Y[:,k1])+(1/20)*np.ptp(Y[:,k1]))
        axs[k2-1,k1].set_ylim(np.min(Y[:,k2])-(1/20)*np.ptp(Y[:,k2]),
                              np.max(Y[:,k2])+(1/20)*np.ptp(Y[:,k2]))
        axs[k2-1,k1].set_xlabel(hdr[k1], fontsize=16)
        axs[k2-1,k1].set_ylabel(hdr[k2], fontsize=16)
        axs[k2-1,k1].tick_params(axis='both', labelsize=10)
        if k1 == 0 and k2 == 1:
            handles, labels = axs[k2-1,k1].get_legend_handles_labels()
            axs[k2-1,k1].set_title('Data Set', fontsize=16, fontweight='bold')
axs[0,1].legend(handles, labels, loc='lower left')
axs[1,2].legend(handles, labels, loc='lower left')
axs[0,1].axis('off')
axs[1,2].axis('off')

# confusion matrices
for h in range(len(xt)):
    for g in range(2):
        if g == 0:
            CM = MBC[h].evaluate('CM')
        if g == 1:
            CM = np.array([[np.mean(xp[h][xt[h]==j1]==j2)
                            for j1 in range(1,nC[h]+1)]
                            for j2 in range(1,nC[h]+1)])
        no = mpl.colors.Normalize(vmin=0, vmax=2*(1/nC[h]))
        sm = mpl.cm.ScalarMappable(norm=no, cmap=cmo)
        axs[h,3+g].matshow(CM, aspect='equal', cmap=cmo, norm=no)
        axs[h,3+g].xaxis.set_ticks_position('bottom')
        axs[h,3+g].set_xticks(range(nC[h]), labels=['{}'.format(xC[i-1]) for i in iC[h]])
        axs[h,3+g].set_yticks(range(nC[h]), labels=['{}'.format(xC[i-1]) for i in iC[h]])
        axs[h,3+g].set_xlabel('true class', fontsize=16)
        axs[h,3+g].set_ylabel('predicted class', fontsize=16)
        axs[h,3+g].tick_params(axis='both', labelsize=10)
        if g == 0:
            axs[h,3+g].set_title('MBC: {} classes'.format(nC[h]),
                                 fontsize=16, fontweight='bold')
        if g == 1:
            axs[h,3+g].set_title('SVC: {} classes'.format(nC[h]),
                                 fontsize=16, fontweight='bold')
        fig.colorbar(sm, ax=axs[h,3+g], orientation='vertical', label='')
        for c1 in range(nC[h]):
            for c2 in range(nC[h]):
                axs[h,3+g].text(c1, c2, '{:.2f}'.format(CM[c2,c1]),
                                fontsize=10, ha='center', va='center')

# classification accuracies
axs[0,2].bar(np.arange(3)-1.5*dx, CA[0,:],
             width=2*dx, align='center', color=(0,0,1), edgecolor='k', label='MBC')
axs[0,2].bar(np.arange(3)+1.5*dx, CA[1,:],
             width=2*dx, align='center', color=(1,0,0), edgecolor='k', label='SVC')
axs[0,2].plot([-1, 0.5, 0.5, 1.5, 1.5, 3], [1/5, 1/5, 1/3, 1/3, 1/2, 1/2], ':k',
              linewidth=2, label='chance')
axs[0,2].axis([(0-1), len(nC), 0, 1])
axs[0,2].set_xticks(range(3), labels=['{}'.format(n) for n in nC])
handles, labels = axs[0,2].get_legend_handles_labels(); o = [1,2,0]
axs[0,2].legend([handles[i] for i in o], [labels[i] for i in o],
                loc='upper left', fontsize=10)
axs[0,2].set_xlabel('number of classes', fontsize=16)
axs[0,2].set_ylabel('classification accuracy', fontsize=16)
axs[0,2].set_title('Classification', fontsize=16, fontweight='bold')
axs[0,2].tick_params(axis='both', labelsize=10)

# enable tight layout
fig.tight_layout()
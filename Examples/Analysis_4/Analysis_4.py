# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Analysis 4: brain age prediction (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 21/02/2022, 20:29: first version in MATLAB
- 27/02/2025, 18:01: ported code to Python (Step 1 & 2)
- 28/02/2025, 11:16: ported code to Python (Step 3)
- 05/03/2025, 11:04: added to GitHub repository
- 05/03/2025, 18:45: unified tick font sizes
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
import scipy as sp
from sklearn import svm
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: load data #########################################################

# load data
train = sp.io.loadmat('PAC_specify.mat')
test  = sp.io.loadmat('PAC_specify_test_age.mat')

# assemble data (MBR)
n1  = train['sID1'].size                    # number of data points
n2  = train['sID2'].size
V1  = np.eye(n1)                            # observation covariances
V2  = np.eye(n2)
YA1 = np.c_[train['GM1'], train['WM1']]     # data matrices
YA2 = np.c_[train['GM2'], train['WM2']]
x1  = np.squeeze(train['y1'])               # label vectors
x2  = np.squeeze(test['y2'])
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

# estimate gamma distribution
x1_min = np.min(x1)-0.5
ab_est = sp.stats.gamma.fit(x1-x1_min, floc=0)

# compute training histogram
xp = np.arange(x_min, x_max+1, 1)
fb = np.concatenate((xp-0.5, np.array([xp[-1]+0.5])))
f1 = np.histogram(x1, bins=fb)[0]
del fb

# define priors for MBR
priors = ['uniform', 'data-driven', 'fitted']
prior  = [{} for pr in priors]
prior[0] = {'x': xp, 'p': (1/np.ptp(xp)) * np.ones(xp.shape)}
prior[1] = {'x': xp, 'p': (1/np.trapz(f1, xp)) * f1}
prior[2] = {'x': xp, 'p': sp.stats.gamma.pdf(xp-x1_min, ab_est[0], loc=0, scale=ab_est[2])}

# Analysis 1: MBR with site/sex as covariates
print('\n-> MBR: train, ', end='')
MBA1 = MBI.model(YA1, x1, X=XA1, V=V1, mb_type='MBR').train()
PP2  = [None for pr in priors]
xMAP = np.zeros((n2,len(priors)))
rA   = np.zeros(len(priors))
maeA = np.zeros(len(priors))
print('test: prior ', end='')
for h in range(len(prior)):
    print('{}, '.format(h+1), end='')
    PP2[h]  = MBI.model(YA2, x2, X=XA2, V=V2, mb_type='MBR').test(MBA1, prior[h])
    for i in range(n2):
        xMAP[i,h] = prior[h]['x'][np.argmax(PP2[h][i,:])]
    rA[h]   = np.corrcoef(xMAP[:,h], x2)[0,1]
    maeA[h] = np.mean(np.abs(xMAP[:,h]-x2))
print('done.')

# Analysis 2: SVR with site/sex as features
print('\n-> SVR: train, ', end='')
SVM1 = svm.SVR(kernel='linear', C=1)
SVM1.fit(YB1, x1)
print('test, ', end='')
xp2  = SVM1.predict(YB2)
rB   = np.corrcoef(xp2, x2)[0,1]
maeB = np.mean(np.abs(xp2-x2))
print('done.\n')

# calculate histograms
nb1 = np.histogram(x1, bins=xb)[0]
nb2 = np.histogram(x2, bins=xb)[0]
nbA = np.zeros((len(priors), xb.size-1))
nbB = np.histogram(xp2, bins=xb)[0]
for h in range(len(prior)):
    nbA[h,:] = np.histogram(xMAP[:,h], bins=xb)[0]


### Step 3: visualize results #################################################

# open figure
fig1 = plt.figure(figsize=(16,10))
axs  = fig1.subplots(3,5)

# 1st row
axs[0,0].bar(xb[:-1]+dx/2, nb1,
             width=dx, align='center', color=(3/4,3/4,3/4), edgecolor='k')
axs[0,0].plot(prior[2]['x'], prior[2]['p']*n1*dx, '-k', linewidth=2)
axs[0,0].axis([x_min, x_max, 0, (11/10)*np.max(nb1)])
axs[0,0].set_xlabel('chronological age [yrs]', fontsize=16)
axs[0,0].set_ylabel('number of subjects', fontsize=16)
axs[0,0].set_title('Training Set', fontsize=16, fontweight='bold')
axs[0,0].tick_params(axis='both', labelsize=10)

for h in range(len(prior)):
    axs[0,1+h].plot([x_min, x_max], [x_min, x_max], '-k', linewidth=1)
    axs[0,1+h].plot(x2, xMAP[:,h], '.b', markersize=5)
    axs[0,1+h].axis([x_min, x_max, x_min, x_max])
    axs[0,1+h].set_aspect('equal', adjustable='box')
    axs[0,1+h].set_xlabel('actual age', fontsize=16)
    axs[0,1+h].set_ylabel('predicted age', fontsize=16)
    if h == 1:
        axs[0,1+h].set_title('MBR with site/gender as covariates',
                             fontsize=16, fontweight='bold')
    axs[0,1+h].tick_params(axis='both', labelsize=10)
    axs[0,1+h].text(x_min+5, x_max-5, 'r = {:.2f}, MAE = {:.2f}'.format(rA[h], maeA[h]),
                    fontsize=12, ha='left', va='center')

axs[0,4].plot([x_min, x_max], [x_min, x_max], '-k', linewidth=1)
axs[0,4].plot(x2, xp2, '.r', markersize=5)
axs[0,4].axis([x_min, x_max, x_min, x_max])
axs[0,4].set_aspect('equal', adjustable='box')
axs[0,4].set_xlabel('actual age', fontsize=16)
axs[0,4].set_ylabel('predicted age', fontsize=16)
axs[0,4].set_title('SVR with site/gender as features',
                   fontsize=16, fontweight='bold')
axs[0,4].tick_params(axis='both', labelsize=10)
axs[0,4].text(x_min+5, x_max-5, 'r = {:.2f}, MAE = {:.2f}'.format(rB, maeB),
              fontsize=12, ha='left', va='center')

# 2nd row
axs[1,0].bar(xb[:-1]+dx/2, nb2,
             width=dx, align='center', color=(3/4,3/4,3/4), edgecolor='k')
axs[1,0].axis([x_min, x_max, 0, (11/10)*np.max(nb2)])
axs[1,0].set_xlabel('chronological age [yrs]', fontsize=16)
axs[1,0].set_ylabel('number of subjects', fontsize=16)
axs[1,0].set_title('Validation Set', fontsize=16, fontweight='bold')
axs[1,0].tick_params(axis='both', labelsize=10)

for h in range(len(prior)):
    axs[1,1+h].bar(xb[:-1]+dx/2, nbA[h,:],
                   width=dx, align='center', color=(0,0,1), edgecolor='k')
    axs[1,1+h].axis([x_min, x_max, 0, (11/10)*np.max(nbA[h,:])])
    axs[1,1+h].set_xlabel('predicted age', fontsize=16)
    axs[1,1+h].set_ylabel('number of subjects', fontsize=16)
    if h == 1:
        axs[1,1+h].set_title('MBR: prediction distribution',
                             fontsize=16, fontweight='bold')
    axs[1,1+h].tick_params(axis='both', labelsize=10)

axs[1,4].bar(xb[:-1]+dx/2, nbB,
             width=dx, align='center', color=(1,0,0), edgecolor='k')
axs[1,4].axis([x_min, x_max, 0, (11/10)*np.max(nbB)])
axs[1,4].set_xlabel('predicted age', fontsize=16)
axs[1,4].set_ylabel('number of subjects', fontsize=16)
axs[1,4].set_title('SVR: prediction distribution',
                   fontsize=16, fontweight='bold')
axs[1,4].tick_params(axis='both', labelsize=10)

# 3rd row
for h in range(len(prior)):
    axs[2,1+h].plot(prior[h]['x'], prior[h]['p'], '-b', linewidth=1)
    axs[2,1+h].axis([x_min, x_max, 0, (11/10)*np.max(prior[h]['p'])])
    if h == 0: axs[2,1+h].set_ylim([0, 2*np.max(prior[h]['p'])])
    axs[2,1+h].set_xlabel('chronological age [yrs]', fontsize=16)
    axs[2,1+h].set_ylabel('prior density', fontsize=16)
    axs[2,1+h].set_title('{} prior'.format(priors[h]),
                         fontsize=16, fontweight='bold')
    axs[2,1+h].tick_params(axis='both', labelsize=10)
axs[2,0].axis('off')
axs[2,4].axis('off')

# open figure
fig2 = plt.figure(figsize=(16,10))
axs  = fig2.subplots(3,5)

# all rows
for h in range(len(prior)):
    for i in range(5):
        axs[h,i].plot(x2[i], (1/10)*np.max(PP2[h][i,:]), 'xk',
                      markersize=7.5, linewidth=3, label='true')
        axs[h,i].plot(xMAP[i,h], (1/10)*np.max(PP2[h][i,:]), '.b',
                      markersize=15, label='MAP')
        axs[h,i].plot(prior[h]['x'], PP2[h][i,:], '-b',
                      linewidth=1)
        axs[h,i].axis([np.min(prior[h]['x']), np.max(prior[h]['x']),
                       0, (11/10)*np.max(PP2[h][i,:])])
        if h == 0 and i == 0:
            axs[h,i].legend(loc='upper right')
        if h == 2:
            axs[h,i].set_xlabel('chronological age [yrs]', fontsize=16)
        if i == 1:
            axs[h,i].set_ylabel('posterior density', fontsize=16)
        if i == 0:
            axs[h,i].set_ylabel('{} prior'.format(priors[h]),
                                fontsize=20, fontweight='bold')
        if h == 0:
            axs[h,i].set_title('Subject {}'.format(i+1),
                               fontsize=20, fontweight='bold')
        axs[h,i].tick_params(axis='both', labelsize=10)

# enable tight layout
fig1.tight_layout()
fig2.tight_layout()
# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Simulation 4: continuous prediction (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 20/02/2022, 22:38: first version in MATLAB
- 21/04/2022, 15:55: ported code to Python
- 06/07/2022, 16:21: modified MBI import
- 20/02/2025, 17:44: added results visualization
- 05/03/2025, 11:00: added to GitHub repository
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
import scipy.linalg as sp_linalg
from sklearn import svm
import matplotlib.pyplot as plt
import matplotlib.gridspec as grs

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
xm = 1
x  = np.random.uniform(-xm, +xm, size=(n,1))
X  = np.c_[x, np.ones(x.shape)]
x  = np.squeeze(x)

# histogram of targets
lim = 1.5
dx  = 0.1
nx  = int((2*lim)/dx)+1
xb  = np.linspace(-lim, +lim, nx)
nb  = np.histogram(x, bins=xb)[0]


### Step 2: generate & analyze data ###########################################

# generate data
B = np.random.normal(mu, np.sqrt(sb), size=(X.shape[1],v))
E = MBI.matnrnd(np.zeros((n,v)), s2*V, Si, 1)
Y = X @ B + E

# preallocate results
r   = np.zeros(2)
MAE = np.zeros(2)

# prepare prior
dx    = 0.01
ns    = int((2*xm)/dx)+1
prior = {'x': np.linspace(-xm, +xm, ns),
         'p': 1/(2*xm)*np.ones(ns)}

# Analysis 1: MBR
MBR = MBI.cvMBI(Y, x, V=V, mb_type='MBR')
MBR.crossval(k=k, cv_mode='kf')
MBR.predict(prior)
r[0]   = MBR.evaluate('r')
MAE[0] = MBR.evaluate('MAE')

# Analysis 2: SVM
xp  = np.zeros(n)
for g in range(k):
    # get training and test set
    i1  = np.array(np.nonzero(MBR.CV[:,g]==1)[0], dtype=int)
    i2  = np.array(np.nonzero(MBR.CV[:,g]==2)[0], dtype=int)
    Y1  = Y[i1,:]
    Y2  = Y[i2,:]
    x1  = x[i1]
    # train and test using SVR
    SVR = svm.SVR(kernel='linear', C=1)
    SVR.fit(Y1, x1)
    xp[i2] = SVR.predict(Y2)
r[1]   = np.corrcoef(xp, x)[0,1]
MAE[1] = np.mean(np.abs(xp-x))


### Step 3: visualize results #################################################

# prepare plotting
x_MBR  = xm
x_SVR  = np.max(np.abs(xp))
nb_MBR = np.histogram(MBR.xp, bins=xb)[0]
nb_SVR = np.histogram(xp, bins=xb)[0]

# open figure
fig = plt.figure(figsize=(16,10))
gsm = grs.GridSpec(2, 3, figure=fig)
axs = [[fig.add_subplot(gsm[i,j]) for j in range(3)] for i in range(2)]
axs = np.array(axs)

# plot labels
axs[0,0].bar(xb[:-1]+dx/2, nb,
             width=8*dx, align='edge', color=(3/4,3/4,3/4), edgecolor='k')
axs[0,0].axis([-lim, +lim, 0, (11/10)*np.max(nb)])
axs[0,0].set_xlabel('target value', fontsize=16)
axs[0,0].set_ylabel('number of samples', fontsize=16)
axs[0,0].set_title('Training Data', fontsize=20, fontweight='bold')
axs[0,0].tick_params(axis='both', labelsize=12)
axs[0,0].text(0, -(2.5/10)*np.max(nb), 'MBR: posterior distributions',
              fontsize=16, fontweight='bold', ha='center', va='center')

# plot MBR predictions
axs[0,1].plot([-x_SVR, +x_SVR], [-x_SVR, +x_SVR], '-k', linewidth=1)
axs[0,1].plot(x, MBR.xp, '.b', markersize=5)
axs[0,1].axis([-x_SVR, +x_SVR, -x_SVR, +x_SVR])
axs[0,1].set_aspect('equal', adjustable='box')
axs[0,1].set_xlabel('actual target values', fontsize=16)
axs[0,1].set_ylabel('predicted target values', fontsize=16)
axs[0,1].set_title('MBR: MAP estimates', fontsize=20, fontweight='bold')
axs[0,1].tick_params(axis='both', labelsize=12)
axs[0,1].text(-x_MBR, +x_MBR, 'r = {:.2f}, MAE = {:.2f}'.format(r[0], MAE[0]), 
              fontsize=12, ha='left', va='bottom')

# plot SVR predictions
axs[0,2].plot([-x_SVR, +x_SVR], [-x_SVR, +x_SVR], '-k', linewidth=1)
axs[0,2].plot(x, xp, '.r', markersize=5)
axs[0,2].axis([-x_SVR, +x_SVR, -x_SVR, +x_SVR])
axs[0,2].set_aspect('equal', adjustable='box')
axs[0,2].set_xlabel('actual target values', fontsize=16)
axs[0,2].set_ylabel('predicted target values', fontsize=16)
axs[0,2].set_title('SVR: SVM estimates', fontsize=20, fontweight='bold')
axs[0,2].tick_params(axis='both', labelsize=12)
axs[0,2].text(-x_MBR, +x_MBR, 'r = {:.2f}, MAE = {:.2f}'.format(r[1], MAE[1]), 
              fontsize=12, ha='left', va='bottom')

# plot MBR histogram
axs[1,1].bar(xb[:-1]+dx/2, nb_MBR,
             width=8*dx, align='edge', color='b', edgecolor='k')
axs[1,1].axis([-lim, +lim, 0, (11/10)*np.max(nb_MBR)])
axs[1,1].set_xlabel('target value', fontsize=16)
axs[1,1].set_ylabel('number of predictions', fontsize=16)
axs[1,1].set_title('MBR: prediction distribution', fontsize=16, fontweight='bold')
axs[1,1].tick_params(axis='both', labelsize=12)

# plot SVR histogram
axs[1,2].bar(xb[:-1]+dx/2, nb_SVR,
             width=8*dx, align='edge', color='r', edgecolor='k')
axs[1,2].axis([-lim, +lim, 0, (11/10)*np.max(nb_SVR)])
axs[1,2].set_xlabel('target value', fontsize=16)
axs[1,2].set_ylabel('number of predictions', fontsize=16)
axs[1,2].set_title('SVR: prediction distribution', fontsize=16, fontweight='bold')
axs[1,2].tick_params(axis='both', labelsize=12)

# plot MBR posteriors
fig.delaxes(axs[1,0])
gsn = grs.GridSpecFromSubplotSpec(2, 2, subplot_spec=gsm[1,0])
a21 = [fig.add_subplot(gsn[i,j]) for (i,j) in [(0,0),(0,1),(1,0),(1,1)]]
for h in range(4):
    a21[h].plot(MBR.xt[h], (1/10)*np.max(MBR.PP[h,:]), 'xk',
                markersize=7.5, linewidth=3, label='true')
    a21[h].plot(MBR.xp[h], (1/10)*np.max(MBR.PP[h,:]), '.b',
                markersize=15, label='mode')
    a21[h].plot(prior['x'], MBR.PP[h,:], '-b',
                linewidth=1)
    a21[h].axis([-x_MBR, +x_MBR, 0, (11/10)*np.max(MBR.PP[h,:])])
    if h == 2:
        a21[h].legend(loc='upper right', fontsize=12)
    if h == 2:
        a21[h].set_xlabel('target value', fontsize=16)
        a21[h].set_ylabel('posterior density', fontsize=16)

# enable tight layout
fig.tight_layout()
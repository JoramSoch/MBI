# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Simulation 3: three-class classification (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 20/02/2022, 12:23: first version in MATLAB
- 19/02/2025, 16:38: ported code to Python (Steps 1 & 2)
- 20/02/2025, 15:02: ported code to Python (Steps 2 & 3)
- 05/03/2025, 10:58: added to GitHub repository
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: specify ground truth & model ######################################

# set ground truth
np.random.seed(2)
mu = 1                                      # class means
Si = np.array([[1,0.5],[0.5,1]])            # covariance structure
s2 = 4                                      # noise variance
n  = 300
k  = 10
v  = 2
C  = 3

# generate classes
x  = np.kron(np.arange(C).reshape((C,1))+1, np.ones((int(n/C),1)))
x  = np.random.permutation(x[:,0])
X  = np.zeros((n,C))
V  = np.eye(n)
for i in range(n):
    X[i,int(x[i]-1)] = 1


### Step 2: generate & analyze the data #######################################

# generate data
B = np.array([[  -mu,   +mu],
              [+2*mu, +2*mu],
              [  +mu,   -mu]])
E = MBI.matnrnd(np.zeros((n,v)), s2*V, Si, 1)
E = np.squeeze(E)
Y = X @ B + E

# preallocate results
CA  = np.zeros(2)
CAp = np.zeros(C)

# prepare prediction grid
lim = 6
dxy = 0.05
n2  = int((2*lim)/dxy)
xy  = np.linspace(-lim+dxy/2, +lim-dxy/2, n2)
x2  = np.concatenate((np.arange(C)+1, np.ones(n2-C)))

# Analysis 1: MBC
MBC   = MBI.cvMBI(Y, x, V=V, mb_type='MBC')
MBC.crossval(k=k, cv_mode='kfc')
MBC.predict()
CA[0] = MBC.evaluate('CA')
MBA   = MBI.model(Y, x, V=V, mb_type='MBC').train()

# Analysis 2: SVC
xp  = np.zeros(n)
for g in range(k):
    # get training and test set
    i1  = np.array(np.nonzero(MBC.CV[:,g]==1)[0], dtype=int)
    i2  = np.array(np.nonzero(MBC.CV[:,g]==2)[0], dtype=int)
    Y1  = Y[i1,:]
    Y2  = Y[i2,:]
    x1  = x[i1]
    # train and test using SVC
    SVC = svm.SVC(kernel='linear', C=1)
    SVC.fit(Y1, x1)
    xp[i2] = SVC.predict(Y2)
CA[1] = np.mean(xp==x)
SVM   = svm.LinearSVC(C=1); SVM.fit(Y, x)

# Analysis 1: priors
for j in range(C):
    prior  = {'x': np.arange(C)+1, 'p': 1/6*np.ones(C)}
    prior['p'][j] = 2/3 
    MBC    = MBI.cvMBI(Y, x, V=V, mb_type='MBC')
    MBC.crossval(k=k, cv_mode='kfc')
    MBC.predict(prior=prior)
    CAp[j] = MBC.evaluate('CA')
    del MBC

# Analysis 1 & 2: predictions
PP  = np.zeros((xy.size,xy.size,3))
Xp  = np.zeros((xy.size,xy.size,3))
PPp = np.zeros((xy.size,xy.size,3,C))
print('\n-> Prediction grid:', end='')
for i in range(xy.size):
    
    # specify test data
    if i % 10 == 0:
        print('\n   - x = ', end='')
    print('{:.3f}, '.format(xy[i]), end='')
    Y2 = np.c_[xy[i]*np.ones((xy.size,1)), xy]
    
    # MBC: posterior probabilities
    pp = MBI.model(Y2, x2, V=np.eye(xy.size), mb_type='MBC').test(MBA)
    for j in range(C):
        PP[:,i,j] = pp[:,j]
    
    # SVC: predicted classes
    xp = SVM.predict(Y2)
    for j in range(C):
        Xp[xp==(j+1),i,j] = 1
    
    # MBC: modified priors
    prior = {'x': np.arange(C)+1, 'p': 1/3*np.ones(C)}
    for j1 in range(C):
        prior['p']     = 1/6*np.ones(C)
        prior['p'][j1] = 2/3
        pp = MBI.model(Y2, x2, V=np.eye(xy.size), mb_type='MBC').test(MBA, prior)
        for j2 in range(C):
            PPp[:,i,j2,j1] = pp[:,j2]

del pp, xp, prior

# edit posterior probabilities
thr =  0.0031308
PP  = 12.92*PP*(PP<=thr)   + 1.055*np.power(PP, 1/2.4)*(PP>thr)
PPp = 12.92*PPp*(PPp<=thr) + 1.055*np.power(PPp, 1/2.4)*(PPp>thr)
PP  = PP*(PP<=1) + 1*(PP>1)
PPp = PPp*(PPp<=1) + 1*(PPp>1)
# Source: https://en.wikipedia.org/w/index.php?title=SRGB&oldid=1226800876#From_CIE_XYZ_to_sRGB


### Step 3: visualize results #################################################

# open figure
fig = plt.figure(figsize=(16,10))
axs = fig.subplots(2,3)

# plot features
cols = [(1,0,0), (0,1,0), (0,0,1)]
for j in range(C):
    axs[0,0].plot(Y[x==(j+1),0], Y[x==(j+1),1], '.',
                  markersize=5, color=cols[j], label='class {}'.format(j+1))
axs[0,0].axis([-lim, +lim, -lim, +lim])
axs[0,0].set_aspect('equal', adjustable='box')
axs[0,0].legend(loc='lower right', fontsize=12)
axs[0,0].set_xlabel('feature 1', fontsize=16)
axs[0,0].set_ylabel('feature 2', fontsize=16)
axs[0,0].set_title('Training Data', fontsize=20, fontweight='bold')
axs[0,0].tick_params(axis='both', labelsize=12)

# plot MBC posterior probabilities
axs[0,1].imshow(PP, extent=[-lim, +lim, -lim, +lim],
                origin='lower', aspect='equal')
axs[0,1].axis([-lim, +lim, -lim, +lim])
axs[0,1].set_xlabel('feature 1', fontsize=16)
axs[0,1].set_ylabel('feature 2', fontsize=16)
axs[0,1].set_title('MBC: posterior probabilities', fontsize=20, fontweight='bold')
axs[0,1].tick_params(axis='both', labelsize=12)
axs[0,1].text(+(9/10)*lim, -(9/10)*lim, 'CA = {:2.2f} %'.format(CA[0]*100),
              color=(1,1,1), fontsize=12, ha='right', va='bottom')

# plot SVC predicted classes
axs[0,2].imshow(Xp, extent=[-lim, +lim, -lim, +lim],
                origin='lower', aspect='equal')
axs[0,2].axis([-lim, +lim, -lim, +lim])
axs[0,2].set_xlabel('feature 1', fontsize=16)
axs[0,2].set_ylabel('feature 2', fontsize=16)
axs[0,2].set_title('SVC: predicted classes', fontsize=20, fontweight='bold')
axs[0,2].tick_params(axis='both', labelsize=12)
axs[0,2].text(+(9/10)*lim, -(9/10)*lim, 'CA = {:2.2f} %'.format(CA[1]*100),
              color=(1,1,1), fontsize=12, ha='right', va='bottom')

# plot MBC with modified priors
for j in range(C):
    axs[1,j].imshow(PPp[:,:,:,j], extent=[-lim, +lim, -lim, +lim],
                    origin='lower', aspect='equal')
    axs[1,j].axis([-lim, +lim, -lim, +lim])
    axs[1,j].set_xlabel('feature 1', fontsize=16)
    axs[1,j].set_ylabel('feature 2', fontsize=16)
    axs[1,j].set_title('MBC: class {} more likely a priori'.format(j+1),
                       fontsize=16, fontweight='bold')
    axs[1,j].tick_params(axis='both', labelsize=12)
    axs[1,j].text(+(9/10)*lim, -(9/10)*lim, 'CA = {:2.2f} %'.format(CAp[0]*100),
                  color=(1,1,1), fontsize=12, ha='right', va='bottom')

# enable tight layout
fig.tight_layout()
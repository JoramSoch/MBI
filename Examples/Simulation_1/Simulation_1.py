# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Simulation 1: two-class classification (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 20/02/2022, 12:23: first version in MATLAB
- 12/04/2022, 19:56: ported code to Python
- 06/07/2022, 16:16: modified MBI import
- 18/02/2025, 17:46: added results visualization
- 05/03/2025, 10:54: added to GitHub repository
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: specify ground truth & model ######################################

# set ground truth
np.random.seed(2)
mu = np.linspace(0, 1, 5)                   # class means
Si = np.array([[1,0.5],[0.5,1]])            # covariance structure
s2 = np.square(np.linspace(1, 5, 5))        # noise variance
n  = 250
k  = 10
v  = 2
C  = 2

# generate classes
x  = np.kron(np.arange(C).reshape((C,1))+1, np.ones((int(n/C),1)))
x  = np.random.permutation(x[:,0])
X  = np.zeros((n,C))
V  = np.eye(n)
for i in range(n):
    X[i,int(x[i]-1)] = 1


### Step 2: generate & analyze the data #######################################

# preallocate results
MBC = [[None for h in range(mu.size)] for g in range(2)]
DA  = np.zeros((2,mu.size))

# run simulations
for h in range(mu.size):

    # generate signals (variance fixed)
    B = np.array([[-mu[h], +mu[h]], [+mu[h], -mu[h]]])
    E = MBI.matnrnd(np.zeros((n,v)), s2[0]*V, Si, 1)
    E = np.squeeze(E)
    Y = X @ B + E
    
    # cross-validated MBC
    MBC[0][h] = MBI.cvMBI(Y, x, V=V, mb_type='MBC')
    MBC[0][h].crossval(k=k, cv_mode='kfc')
    MBC[0][h].predict()
    DA[0,h]   = MBC[0][h].evaluate('CA')
    
    # generate signals (distance fixed)
    B = np.array([[-mu[-1], +mu[-1]], [+mu[-1], -mu[-1]]])
    E = MBI.matnrnd(np.zeros((n,v)), s2[h]*V, Si, 1)
    E = np.squeeze(E)
    Y = X @ B + E
    
    # cross-validated MBC
    MBC[1][h] = MBI.cvMBI(Y, x, V=V, mb_type='MBC')
    MBC[1][h].crossval(k=k, cv_mode='kfc')
    MBC[1][h].predict()
    DA[1,h]   = MBC[1][h].evaluate('CA')

    
### Step 3: visualize results #################################################

# create colormap
dc   = 0.01
ns   = int(1/dc)
cmap = np.r_[np.c_[np.linspace(0, 1-dc, ns), np.zeros((ns,1)), np.ones((ns,1))],
             np.array([[1, 0, 1]]),
             np.c_[np.ones((ns,1)), np.zeros((ns,1)), np.linspace(1-dc, 0, ns)]]
lims = [4, 12]

# open figure
fig = plt.figure(figsize=(21,7))
axs = fig.subplots(2, mu.size+1)

# plot results
for g in range(2):
    
    # plot features
    for h in range(mu.size):
        for i in range(n):
            ind = round(MBC[g][h].PP[i,1]*(2*ns))
            axs[g,h].plot(MBC[g][h].Y[i,0], MBC[g][h].Y[i,1], '.', 
                          color=(cmap[ind,0], cmap[ind,1], cmap[ind,2]))
        axs[g,h].axis([-lims[g], +lims[g], -lims[g], +lims[g]])
        axs[g,h].set_aspect('equal', adjustable='box')
        if g == 1 and h == 0:
            axs[g,h].set_xlabel('feature 1', fontsize=16)
            axs[g,h].set_ylabel('feature 2', fontsize=16)
        if g == 0:
            axs[g,h].set_title('distance: {:.2f}'.format(np.sqrt(8*(mu[h])**2)),
                               fontsize=20, fontweight='bold')
        if g == 1:
            axs[g,h].set_title('std. dev.: {:.0f}'.format(np.sqrt(s2[h])),
                               fontsize=20, fontweight='bold')
        axs[g,h].tick_params(axis='both', labelsize=12)
        
    # plot accuracies
    if g == 0: x_gh = np.sqrt(8*(mu)**2)
    if g == 1: x_gh = np.sqrt(s2)
    axs[g,-1].plot(x_gh, DA[g,:], ':ok')
    axs[g,-1].set_xlim((np.min(x_gh)-0.1), (np.max(x_gh)+0.1))
    axs[g,-1].set_ylim((0.5-0.05), (1+0.05))
    axs[g,-1].set_ylabel('classification accuracy', fontsize=16)
    if g == 0:
        axs[g,-1].set_xlabel('distribution distance', fontsize=16)
        axs[g,-1].set_title('std. dev. fixed', fontsize=20, fontweight='bold')
    if g == 1:
        axs[g,-1].set_xlabel('standard deviation', fontsize=16)
        axs[g,-1].set_title('distance fixed', fontsize=20, fontweight='bold')
    axs[g,-1].tick_params(axis='both', labelsize=12)

# enable tight layout
fig.tight_layout()
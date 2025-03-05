# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Simulation 2: two classes with confound (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 20/02/2022, 12:23: first version in MATLAB
- 19/02/2025, 12:38: ported code to Python
- 19/02/2025, 16:03: edited plotting details
- 20/02/2025, 13:49: changed to linear SVC with C=1
- 05/03/2025, 10:56: added to GitHub repository
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import numpy as np
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)


### Step 1: specify ground truth & model ######################################

# set ground truth
np.random.seed(3)
mu = 1                                      # class means
b3 = 2                                      # confound effect
Si = np.array([[1,0.5],[0.5,1]])            # covariance structure
s2 = 4                                      # noise variance
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

# generate confound
c  = 1.5*np.random.uniform(size=(n,1))-0.75
c[x==1] = c[x==1] + 0.25
c[x==2] = c[x==2] - 0.25
X  = np.c_[X, c]


### Step 2: generate & analyze the data #######################################

# generate data
B = np.array([[-mu, +mu],
              [+mu, -mu],
              [  0,  b3]])
E = MBI.matnrnd(np.zeros((n,v)), s2*V, Si, 1)
E = np.squeeze(E)
Y = X @ B + E

# preallocate results
MBC = [None for h in range(2)]
SVC = [None for h in range(2)]
Xp  = [None for h in range(2)]
CA  = np.zeros((2,2))
la  = np.logical_and

# prepare decision boundary
lim = 6
dxy = 0.05
n2  = int((2*lim)/dxy)+1
Y2a = np.c_[-lim*np.ones((n2,1)), np.linspace(-lim, +lim, n2)]
Y2b = np.c_[+lim*np.ones((n2,1)), np.linspace(-lim, +lim, n2)]
x2  = np.concatenate((np.array([1]), 2*np.ones(n2-1)))

# Analysis 1: MBC w/o covariate
MBC[0]  = MBI.cvMBI(Y, x, V=V, mb_type='MBC')
MBC[0].crossval(k=k, cv_mode='kfc')
MBC[0].predict()
CA[0,0] = MBC[0].evaluate('CA')

# Analysis 2: MBC with covariate
MBC[1]  = MBI.cvMBI(Y, x, X=c, V=V, mb_type='MBC')
MBC[1].crossval(k=k, cv_mode='kfc')
MBC[1].predict()
CA[1,0] = MBC[1].evaluate('CA')

# Analysis 3: SVM w/o covariate
# Analysis 4: SVM with prior regression
Xc  = np.c_[c, np.ones((n,1))]
Yr  = (np.eye(n) - Xc @ np.linalg.inv(Xc.T @ Xc) @ Xc.T) @ Y
xp3 = np.zeros(n)
xp4 = np.zeros(n)
for g in range(k):
    # get training and test set
    i1  = np.array(np.nonzero(MBC[0].CV[:,g]==1)[0], dtype=int)
    i2  = np.array(np.nonzero(MBC[0].CV[:,g]==2)[0], dtype=int)
    Y1  = Y[i1,:]
    Y2  = Y[i2,:]
    Yr1 = Yr[i1,:]
    Yr2 = Yr[i2,:]
    x1  = x[i1]
    # train and test using SVC w/o covariate
    SVC[0]  = svm.SVC(kernel='linear', C=1)
    SVC[0].fit(Y1, x1)
    xp3[i2] = SVC[0].predict(Y2)
    # train and test using SVC with prior regression
    SVC[1]  = svm.LinearSVC(C=1)
    SVC[1].fit(Yr1, x1)
    xp4[i2] = SVC[1].predict(Yr2)
del i1, i2, Y1, Y2, Yr1, Yr2, x1
Xp[0]   = xp3
Xp[1]   = xp4
CA[0,1] = np.mean(xp3==x)
CA[1,1] = np.mean(xp4==x)
del xp3, xp4

# Analysis 1: decision boundary
MBA   = MBI.model(Y, x, V=V, mb_type='MBC').train()
PPa   = MBI.model(Y2a, x2, V=np.eye(n2), mb_type='MBC').test(MBA)
PPb   = MBI.model(Y2b, x2, V=np.eye(n2), mb_type='MBC').test(MBA)
ka    = np.argmin(np.abs(PPa[:,0]-PPa[:,1]))
kb    = np.argmin(np.abs(PPb[:,0]-PPb[:,1]))
Y_MBC = np.array([Y2a[ka,:], Y2b[kb,:]])

# Analysis 3: decision boundary
SVM   = svm.LinearSVC(C=1); SVM.fit(Y, x)
xpa   = SVM.predict(Y2a)
xpb   = SVM.predict(Y2b)
ka    = np.argmax(np.abs(np.diff(xpa)))
kb    = np.argmax(np.abs(np.diff(xpb)))
Y_SVC = np.array([np.mean(Y2a[ka:(ka+2),:], axis=0),
                  np.mean(Y2b[kb:(kb+2),:], axis=0)])


### Step 3: visualize results #################################################

# create colormap
viri = plt.get_cmap('viridis')  # viridis color map
cmap = np.array(viri.colors)    # 256 x 3 array of RGB values
nc   = len(cmap)                # scalar mappable on interval [-1,+1]
sm   = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=-1, vmax=+1),
                             cmap=mpl.cm.viridis)

# open figure
fig = plt.figure(figsize=(16,10))
axs = fig.subplots(2,3)

# plot features
axs[0,0].plot(Y[x==1,0], Y[x==1,1], '.r',
              markersize=5, label='class 1')
axs[0,0].plot(Y[x==2,0], Y[x==2,1], 'sb',
              markersize=2, linewidth=2, label='class 2')
axs[0,0].plot([-mu, +mu], [+mu, -mu], 'xk',
              markersize=10, linewidth=3)
axs[0,0].axis([-lim, +lim, -lim, +lim])
axs[0,0].set_aspect('equal', adjustable='box')
axs[0,0].legend(loc='upper right', fontsize=12)
axs[0,0].set_xlabel('feature 1', fontsize=16)
axs[0,0].set_ylabel('feature 2', fontsize=16)
axs[0,0].set_title('Classes', fontsize=20, fontweight='bold')
axs[0,0].tick_params(axis='both', labelsize=12)

# plot confound
for i in range(n):
    ind = round((c[i,0]-np.min(c))/(np.max(c)-np.min(c)) * (nc-1))
    axs[1,0].plot(Y[i,0], Y[i,1], '.', 
                  markersize=5, color=(cmap[ind,0], cmap[ind,1], cmap[ind,2]))
axs[1,0].plot([0, 0, -1/8*b3, 0, +1/8*b3], [0, b3, 6/8*b3, b3, 6/8*b3], '-k', linewidth=2)
axs[1,0].axis([-lim, +lim, -lim, +lim])
axs[1,0].set_aspect('equal', adjustable='box')
axs[1,0].set_xlabel('feature 1', fontsize=16)
axs[1,0].set_ylabel('feature 2', fontsize=16)
axs[1,0].set_title('Confound', fontsize=20, fontweight='bold')
axs[1,0].tick_params(axis='both', labelsize=12)
fig.colorbar(sm, ax=axs[1,0], orientation='vertical', label='covariate value')

# plot MBC w/o correction
axs[0,1].plot(Y[la(x==1, MBC[0].xp==1),0], Y[la(x==1, MBC[0].xp==1),1], '.r',
              markersize=5, label='class 1, predicted 1')
axs[0,1].plot(Y[la(x==2, MBC[0].xp==1),0], Y[la(x==2, MBC[0].xp==1),1], 'sr',
              markersize=2, linewidth=2, label='class 2, predicted 1')
axs[0,1].plot(Y[la(x==1, MBC[0].xp==2),0], Y[la(x==1, MBC[0].xp==2),1], '.b',
              markersize=5, label='class 1, predicted 2')
axs[0,1].plot(Y[la(x==2, MBC[0].xp==2),0], Y[la(x==2, MBC[0].xp==2),1], 'sb',
              markersize=2, linewidth=2, label='class 2, predicted 2')
axs[0,1].plot(Y_MBC[:,0], Y_MBC[:,1], '-',
              color=(0.5,0.5,0.5), linewidth=1)
axs[0,1].axis([-lim, +lim, -lim, +lim])
axs[0,1].set_aspect('equal', adjustable='box')
axs[0,1].legend(loc='upper left', fontsize=12)
axs[0,1].set_xlabel('feature 1', fontsize=16)
axs[0,1].set_ylabel('feature 2', fontsize=16)
axs[0,1].set_title('MBC w/o correction', fontsize=20, fontweight='bold')
axs[0,1].tick_params(axis='both', labelsize=12)
axs[0,1].text(+(9/10)*lim, -(9/10)*lim, 'CA = {:2.2f} %'.format(CA[0,0]*100),
              fontsize=12, ha='right', va='bottom')

# plot MBC with covariate inclusion
axs[1,1].plot(Y[la(x==1, MBC[1].xp==1),0], Y[la(x==1, MBC[1].xp==1),1], '.r',
              markersize=5, label='class 1, predicted 1')
axs[1,1].plot(Y[la(x==2, MBC[1].xp==1),0], Y[la(x==2, MBC[1].xp==1),1], 'sr',
              markersize=2, linewidth=2, label='class 2, predicted 1')
axs[1,1].plot(Y[la(x==1, MBC[1].xp==2),0], Y[la(x==1, MBC[1].xp==2),1], '.b',
              markersize=5, label='class 1, predicted 2')
axs[1,1].plot(Y[la(x==2, MBC[1].xp==2),0], Y[la(x==2, MBC[1].xp==2),1], 'sb',
              markersize=2, linewidth=2, label='class 2, predicted 2')
axs[1,1].plot(Y_MBC[:,0], Y_MBC[:,1], '-',
              color=(0.5,0.5,0.5), linewidth=1)
axs[1,1].axis([-lim, +lim, -lim, +lim])
axs[1,1].set_aspect('equal', adjustable='box')
axs[1,1].set_xlabel('feature 1', fontsize=16)
axs[1,1].set_ylabel('feature 2', fontsize=16)
axs[1,1].set_title('MBC with covariate inclusion', fontsize=20, fontweight='bold')
axs[1,1].tick_params(axis='both', labelsize=12)
axs[1,1].text(+(9/10)*lim, -(9/10)*lim, 'CA = {:2.2f} %'.format(CA[1,0]*100),
              fontsize=12, ha='right', va='bottom')

# plot SVC w/o correction
axs[0,2].plot(Y[la(x==1, Xp[0]==1),0], Y[la(x==1, Xp[0]==1),1], '.r',
              markersize=5, label='class 1, predicted 1')
axs[0,2].plot(Y[la(x==2, Xp[0]==1),0], Y[la(x==2, Xp[0]==1),1], 'sr',
              markersize=2, linewidth=2, label='class 2, predicted 1')
axs[0,2].plot(Y[la(x==1, Xp[0]==2),0], Y[la(x==1, Xp[0]==2),1], '.b',
              markersize=5, label='class 1, predicted 2')
axs[0,2].plot(Y[la(x==2, Xp[0]==2),0], Y[la(x==2, Xp[0]==2),1], 'sb',
              markersize=2, linewidth=2, label='class 2, predicted 2')
axs[0,2].plot(Y_SVC[:,0], Y_SVC[:,1], '-',
              color=(0.5,0.5,0.5), linewidth=1)
axs[0,2].axis([-lim, +lim, -lim, +lim])
axs[0,2].set_aspect('equal', adjustable='box')
axs[0,2].set_xlabel('feature 1', fontsize=16)
axs[0,2].set_ylabel('feature 2', fontsize=16)
axs[0,2].set_title('SVC w/o correction', fontsize=20, fontweight='bold')
axs[0,2].tick_params(axis='both', labelsize=12)
axs[0,2].text(+(9/10)*lim, -(9/10)*lim, 'CA = {:2.2f} %'.format(CA[0,1]*100),
              fontsize=12, ha='right', va='bottom')

# plot SVC with prior regression
axs[1,2].plot(Y[la(x==1, Xp[1]==1),0], Y[la(x==1, Xp[1]==1),1], '.r',
              markersize=5, label='class 1, predicted 1')
axs[1,2].plot(Y[la(x==2, Xp[1]==1),0], Y[la(x==2, Xp[1]==1),1], 'sr',
              markersize=2, linewidth=2, label='class 2, predicted 1')
axs[1,2].plot(Y[la(x==1, Xp[1]==2),0], Y[la(x==1, Xp[1]==2),1], '.b',
              markersize=5, label='class 1, predicted 2')
axs[1,2].plot(Y[la(x==2, Xp[1]==2),0], Y[la(x==2, Xp[1]==2),1], 'sb',
              markersize=2, linewidth=2, label='class 2, predicted 2')
axs[1,2].plot(Y_SVC[:,0], Y_SVC[:,1], '-',
              color=(0.5,0.5,0.5), linewidth=1)
axs[1,2].axis([-lim, +lim, -lim, +lim])
axs[1,2].set_aspect('equal', adjustable='box')
axs[1,2].set_xlabel('feature 1', fontsize=16)
axs[1,2].set_ylabel('feature 2', fontsize=16)
axs[1,2].set_title('SVC with prior regression', fontsize=20, fontweight='bold')
axs[1,2].tick_params(axis='both', labelsize=12)
axs[1,2].text(+(9/10)*lim, -(9/10)*lim, 'CA = {:2.2f} %'.format(CA[1,1]*100),
              fontsize=12, ha='right', va='bottom')

# enable tight layout
fig.tight_layout()
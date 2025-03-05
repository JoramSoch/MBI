# -*- coding: utf-8 -*-
"""
Multivariate Bayesian Inversion for Classification and Regression
Analysis 2: MNIST digit recognition (Python script)

Author: Joram Soch, OvGU Magdeburg
E-Mail: joram.soch@ovgu.de

Version History:
- 03/01/2025, 16:57: first version in MATLAB
- 26/02/2025, 13:07: ported code to Python (Step 1 & 2)
- 26/02/2025, 17:03: ported code to Python (Step 3)
- 05/03/2025, 11:02: added to GitHub repository
"""


# specify MBI path
MBI_dir = '../../Python/'

# import packages
import os
import readMNIST
import numpy as np
import scipy.sparse as sp_sparse
import statsmodels.api as sm
from sklearn import svm
import matplotlib as mpl
import matplotlib.pyplot as plt

# import MBI module
orig_dir = os.getcwd()
os.chdir(MBI_dir)
import MBI
os.chdir(orig_dir)

# define steps
steps = [1, 2, 3]


### Step 1: load data #########################################################
if 1 in steps:

    # load MNIST data
    fnY1  = 'train-images.idx3-ubyte'       # training data filenames
    fnx1  = 'train-labels.idx1-ubyte'
    fnY2  = 't10k-images.idx3-ubyte'        # test data filenames
    fnx2  = 't10k-labels.idx1-ubyte'
    MNIST = readMNIST.MnistDataloader(fnY1, fnx1, fnY2, fnx2)
    (Y1, x1), (Y2, x2) = MNIST.load_data()
    
    # clip and scale images
    nxy = 28                                # width and height in pixels
    dxy = 4                                 # pixels to remove on each side
    Y1  = np.array([[[Y1[n][y][x] for n in range(len(Y1))]
                                  for x in range(dxy,nxy-dxy)]
                                  for y in range(dxy,nxy-dxy)])
    Y2  = np.array([[[Y2[n][y][x] for n in range(len(Y2))]
                                  for x in range(dxy,nxy-dxy)]
                                  for y in range(dxy,nxy-dxy)])
    Y1  = Y1/255                            # transform [0,255] to [0,1]
    Y2  = Y2/255
    del MNIST, nxy, dxy
    
    # extract data
    v   = Y1.shape[0]*Y2.shape[1]           # number of features
    n1  = Y1.shape[2]                       # number of training data points
    n2  = Y2.shape[2]                       # number of test data points
    Y1  = np.reshape(Y1,(v,n1),order='F').T # training data matrix
    Y2  = np.reshape(Y2,(v,n2),order='F').T # test data matrix
    x1  = np.array(x1)                      # training labels
    x2  = np.array(x2)                      # test labels
    x1[x1==0] = 10                          # replace 0 by 10
    x2[x2==0] = 10
    
    # save extracted data
    np.savez_compressed('MNIST_data.npz', Y1=Y1, Y2=Y2, x1=x1, x2=x2)


### Step 2: analyze data ######################################################
if 2 in steps:
    
    # load extracted data
    data = np.load('MNIST_data.npz')
    Y1   = data['Y1']
    Y2   = data['Y2']
    x1   = data['x1']
    x2   = data['x2']
    del data
    
    # specify analyses
    N1 = np.concatenate((np.array([1000]), np.arange(2000, x1.size+2000, 2000)))
    N2 = np.array([1000, x2.size])
    
    # preallocate results
    CA_MBC = np.zeros((len(N2),len(N1)))
    CA_SVC = np.zeros((len(N2),len(N1)))
    print('\n-> Train and test on MNIST data set:')
    
    # loop over training data points
    for i in range(len(N1)):
        
        # get number of data points
        n1 = N1[i]
        print('   - n1 = {}:'.format(n1))
        
        # MBC: training
        print('     - training: MBC ... ', end='')
        MBA1 = MBI.model(Y1[:n1,:], x1[:n1], V=sp_sparse.eye(n1), mb_type='MBC').train()
        print('successful!')
        
        # SVC: training
        print('     - training: SVC ... ', end='');
        SVM1 = svm.SVC(kernel='linear', C=1)
        SVM1.fit(Y1[:n1,:], x1[:n1])
        print('successful!')
        
        # loop over test data points
        for j in range(len(N2)):
            
            # get number of data points
            n2 = N2[j]
            print('     - n2 = {}:'.format(n2))
            
            # MBC: testing
            print('       - testing: MBC ... ', end='')
            PP2 = MBI.model(Y2[:n2,:], x2[:n2], V=sp_sparse.eye(n2), mb_type='MBC').test(MBA1)
            xp  = np.argmax(PP2, axis=1) + 1
            CA_MBC[j,i] = np.mean(xp==x2[:n2])
            print('successful!')
            
            # SVC: testing
            print('       - testing: SVC ... ', end='')
            xp2 = SVM1.predict(Y2[:n2,:])
            CA_SVC[j,i] = np.mean(xp2==x2[:n2])
            print('successful!')
            
    # save analysis results
    np.savez_compressed('MNIST_analysis.npz', N1=N1, N2=N2,
                        MBA1=MBA1, PP2=PP2, SVM1=SVM1, xp2=xp2,
                        CA_MBC=CA_MBC, CA_SVC=CA_SVC)


### Step 3: visualize results #################################################
if 3 in steps:
    
    # load extracted data
    data = np.load('MNIST_data.npz')
    Y1   = data['Y1']
    Y2   = data['Y2']
    x1   = data['x1']
    x2   = data['x2']
    n1   = Y1.shape[0]
    nC   = np.max(x2)
    del data
    
    # load analysis results
    analysis = np.load('MNIST_analysis.npz', allow_pickle=True)
    N1       = analysis['N1']
    N2       = analysis['N2']
    n2       = N2[-1]
    x2       = x2[:n2]
    MBA1     = analysis['MBA1'].item()
    PP2      = analysis['PP2']
    xp2      = analysis['xp2']
    CA_MBC   = analysis['CA_MBC']
    CA_SVC   = analysis['CA_SVC']
    del analysis
    
    # create confusion matrices
    iC     = [9] + list(range(9))
    PP_max = np.max(PP2, axis=1)
    xp     = np.argmax(PP2, axis=1) + 1
    CM_MBC = np.array([[np.mean(xp[x2==j1]==j2) for j1 in range(1,nC+1)]
                                                for j2 in range(1,nC+1)])
    CM_SVC = np.array([[np.mean(xp[x2==j1]==j2) for j1 in range(1,nC+1)]
                                                for j2 in range(1,nC+1)])
    CM_MBC = CM_MBC[iC,:][:,iC]
    CM_SVC = CM_SVC[iC,:][:,iC]
    
    # calculate proportions correct
    dx    = 0.04
    alpha = 0.05
    xe_PP = np.arange(0, 1+dx, dx)
    xc_PP = np.arange(0+dx/2, 1+dx/2, dx)
    f_mean= np.zeros(xc_PP.size)
    f_CI  = np.zeros((2,xc_PP.size))
    for j in range(xc_PP.size):
        i_PP = [i for i in range(n2) if  PP_max[i]> xe_PP[j]
                                     and PP_max[i]<=xe_PP[j+1]]
        n_PP = len(i_PP)
        if n_PP > 0:
            n_CA      = np.sum(xp[i_PP]==x2[i_PP])
            p_CI      = sm.stats.proportion_confint(n_CA, n_PP,
                                                    alpha=0.05, method='beta')
            f_mean[j] = n_CA/n_PP
            f_CI[:,j] = np.array(p_CI)
        else:
            f_mean[j] = np.nan
            f_CI[:,j] = np.nan*np.ones(2)
    del dx, xe_PP, i_PP, n_PP, n_CA
    
    # extract precision matrices
    O1 = MBA1['post']['O1']
    L1 = MBA1['post']['L1']
    L1 = L1[iC,:][:,iC]
    
    # create colormap
    cmap = np.r_[np.c_[np.linspace(0, (8/9), 9), np.linspace(0, (8/9), 9), np.linspace(0, (8/9), 9)],
                 np.array([[1, 1, 1]]),
                 np.c_[np.linspace((89/90), 0, 90), np.ones((90,1)), np.linspace((89/90), 0, 90)]]
    cmo  = mpl.colors.ListedColormap(cmap)
    labs = ['{}'.format(i) for i in range(nC)]
    
    # open figure
    fig1 = plt.figure(figsize=(18,10))
    axs  = fig1.subplots(2,3)
    
    # classification accuracies
    axs[0,0].plot(N1, CA_MBC[-1,:], '-b', linewidth=2, label='MBC')
    axs[0,0].plot(N1, CA_SVC[-1,:], '-r', linewidth=2, label='SVC')
    axs[0,0].plot([0, np.max(N1)], [1/nC, 1/nC], ':k', linewidth=2, label='chance')
    axs[0,0].axis([0, np.max(N1), 0, 1])
    axs[0,0].legend(loc='right')
    axs[0,0].set_xlabel('number of training samples', fontsize=16)
    axs[0,0].set_ylabel('classification accuracy', fontsize=16)
    axs[0,0].set_title('Classification', fontsize=16, fontweight='bold')
    
    # confusion matrices
    for g in range(2):
        if g == 0: CM = CM_MBC
        if g == 1: CM = CM_SVC
        no = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(norm=no, cmap=cmo)
        axs[0,1+g].matshow(CM, aspect='equal', cmap=cmo, norm=no)
        axs[0,1+g].xaxis.set_ticks_position('bottom')
        axs[0,1+g].set_xticks(range(nC), labels=labs)
        axs[0,1+g].set_yticks(range(nC), labels=labs)
        axs[0,1+g].set_xlabel('true class', fontsize=16)
        axs[0,1+g].set_ylabel('predicted class', fontsize=16)
        axs[0,1+g].tick_params(axis='both', labelsize=10)
        if g == 0:
            axs[0,1+g].set_title('MBC: {} classes'.format(nC),
                                 fontsize=16, fontweight='bold')
        if g == 1:
            axs[0,1+g].set_title('SVC: {} classes'.format(nC),
                                 fontsize=16, fontweight='bold')
        fig1.colorbar(sm, ax=axs[0,1+g], orientation='vertical', label='')
        for c1 in range(nC):
            for c2 in range(nC):
                axs[0,1+g].text(c1, c2, '{:.2f}'.format(CM[c2,c1]),
                                fontsize=10, ha='center', va='center')
    
    # maximum posterior probability
    axs[1,0].plot(xc_PP, f_mean, 'ob',
                  linewidth=2, markersize=10, markerfacecolor='b', label='average')
    axs[1,0].errorbar(xc_PP, f_mean, yerr=np.array([f_mean-f_CI[0,:], f_CI[1,:]-f_mean]),
                      fmt='none', ecolor='b', elinewidth=2,
                      capsize=8, markeredgewidth=2, label='95% CI')
    axs[1,0].plot([0,1], [0,1], ':k',
                  linewidth=2, label='identity')
    axs[1,0].axis([0, 1, 0, 1])
    handles, labels = axs[1,0].get_legend_handles_labels(); o = [0,2,1]
    axs[1,0].legend([handles[i] for i in o], [labels[i] for i in o],
                    loc='upper left')
    axs[1,0].set_xticks(np.arange(0, 1.1, 0.1))
    axs[1,0].set_yticks(np.arange(0, 1.1, 0.1))
    axs[1,0].set_xlabel('posterior probability of most likely class', fontsize=16)
    axs[1,0].set_ylabel('frequency of most likely being true class', fontsize=14)
    axs[1,0].set_title('Frequency vs. Probability', fontsize=16, fontweight='bold')
    
    # posterior inverse scale matrix
    O1_max = np.max(O1)
    norm   = mpl.colors.Normalize(vmin=-O1_max, vmax=+O1_max)
    axs[1,1].matshow(O1, aspect='equal', norm=norm)
    axs[1,1].xaxis.set_ticks_position('bottom')
    axs[1,1].set_xticks(np.arange(50, Y1.shape[1]+50, 50))
    axs[1,1].set_yticks(np.arange(50, Y1.shape[1]+50, 50))
    axs[1,1].set_xlabel('image pixel', fontsize=16)
    axs[1,1].set_ylabel('image pixel', fontsize=16)
    axs[1,1].set_title('MBC: posterior inverse scale matrix', fontsize=16, fontweight='bold')
    
    # posterior inverse scale matrix
    L1_max = np.max(L1)
    norm   = mpl.colors.Normalize(vmin=-L1_max, vmax=+L1_max)
    axs[1,2].matshow(L1, aspect='equal', norm=norm)
    axs[1,2].xaxis.set_ticks_position('bottom')
    axs[1,2].set_xticks(range(nC), labels=labs)
    axs[1,2].set_yticks(range(nC), labels=labs)
    axs[1,2].set_xlabel('digit category', fontsize=16)
    axs[1,2].set_ylabel('digit category', fontsize=16)
    axs[1,2].set_title('MBC: posterior precision matrix', fontsize=16, fontweight='bold')
    
    # open figure
    fig2 = plt.figure(figsize=(20,8))
    axs  = fig2.subplots(3,nC)
    la   = np.logical_and
    w    = round(np.sqrt(Y1.shape[1]))
    
    # test examples
    for j in range(1,nC+1):
        for g in range(3):
            
            # select image
            if g < 2:
                Y_j  = Y2[la(x2==j,xp==j),:]            # correctly predicted
                xp_j = xp[la(x2==j,xp==j)]              # predicted class
                PP_j = PP_max[la(x2==j,xp==j)]          # posterior probability
            else:
                Y_j  = Y2[la(x2==j,xp!=j),:]            # incorrectly predicted
                xp_j = xp[la(x2==j,xp!=j)]              # predicted class
                PP2j = PP2[la(x2==j,xp!=j),:]           # posterior probabilities
                PP_j = np.array([PP2j[i,xp_j[i]-1] for i in range(Y_j.shape[0])])
                del PP2j
            if g == 0:
                i = np.argmax(PP_j)                     # high-confidence hit
            elif g == 1:
                i = np.argmin(PP_j)                     # low-confidence hit
            elif g == 2:
                i = 0                                   # randomly correct
              # i = np.argmax(PP_j)                     # high-confidence miss
            xp_i = xp_j[i]
            pp_i = PP_j[i]
            Y_i  = np.reshape(Y_j[i,:], (w,w), order='F')
             
            # plot image
            if j < nC: k = j
            else:      k = 0
            lC   = list(range(1,nC)) + [0]
            norm = mpl.colors.Normalize(vmin=0, vmax=1)
            axs[g,k].matshow(Y_i, aspect='equal', norm=norm)
            axs[g,k].xaxis.set_ticks_position('bottom')
            axs[g,k].set_xticks([])
            axs[g,k].set_yticks([])
            if j == nC:
                if g == 0:
                    axs[g,k].set_ylabel('maximum PP',
                                        fontsize=16, fontweight='bold')
                elif g == 1:
                    axs[g,k].set_ylabel('minimum PP',
                                        fontsize=16, fontweight='bold')
                elif g == 2:
                    axs[g,k].set_ylabel('     incorrect prediction',
                                        fontsize=14, fontweight='bold')
            axs[g,k].set_title('PP(\'{}\') = {:.2f}'.format(lC[xp_i-1], pp_i),
                               fontsize=16, fontweight='bold')
    
    # delete variables
    del Y_j, PP_j, xp_i, pp_i, Y_i
    
    # enable tight layout
    fig1.tight_layout()
    fig2.tight_layout()
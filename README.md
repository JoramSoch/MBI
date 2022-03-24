# MBI

### Multivariate Bayesian Inversion for Classification and Regression

These scripts allow to apply multivariate Bayesian inversion (MBI) for classification (MBC) and regression (MBR) to measured multivariate signals ("features") in order to predict distinct categories ("classes") or continuous variables ("targets"), possibly accounting for confounding variables ("covariates"). A manuscript describing this novel machine learning approach to supervised learning is currently in preparation.


## Getting started with MATLAB

> These tools rely on functions from the [cvLME package](https://github.com/JoramSoch/cvLME) (`MGLM_Bayes.m`) and from the [ML4ML toolbox](https://github.com/JoramSoch/ML4ML) (`ML_CV.m`) which for simplicity are currently included in this repository.

Let `Y1` and `Y2` be feature matrices from training and test data and let `x1` and `x2` be natural numbers representing classes in training and test data. Then, the simplest version of MBC can be used to calculate posterior class probabilities in the test data as follows:

```matlab
% multivariate Bayesian classification
MBA1 = mbitrain(Y1, x1, [], [], 'MBC');
PP2  = mbitest(Y2, x2, [], [], MBA1, []);
```

Let `Y1` and `Y2` be feature matrices from training and test data and let `x1` and `x2` be real numbers representing targets in training and test data. Then, the simplest version of MBR can be used to calculate posterior target densities in the test data as follows:

```matlab
% multivariate Bayesian regression
MBA1 = mbitrain(Y1, x1, [], [], 'MBR');
PP2  = mbitest(Y2, x2, [], [], MBA1, []);
```

The output `PP2` is a matrix of posterior probabilities in the test data, with one row corresponding to one data point and each row either representing discrete probability masses or a continuous probability density.

Let `Y` be an entire feature matrix and let `x` be an entire vector of class labels or target values across all data points. Then, cross-validated MBC or MBR (number of CV folds `k`) can be easily implemented as follows:

```matlab
% cross-validated MBC
CV  = ML_CV(x, k, 'kfc');
MBC = ML_MBI(Y, x, [], [], CV, 'MBC', []);
```

```matlab
% cross-validated MBR
CV  = ML_CV(numel(x), k, 'kf');
MBR = ML_MBI(Y, x, [], [], CV, 'MBR', []);
```

For extra input parameters of `ML_MBI.m` and fields of the output structure `MBC`/`MBR`, type `help ML_MBI` into the command window. For manipulating the prior probabilities for prediction in the test data, type `help mbitest`.


## Getting started with Python

> These tools rely on the [cvLME package](https://github.com/JoramSoch/cvLME) (`cvBMS.py`) which is currently included in the repository.

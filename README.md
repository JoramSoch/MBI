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
CV  = ML_CV(x, k, 'kfc');                  % k-folds on points per class
MBC = ML_MBI(Y, x, [], [], CV, 'MBC', []);
DA  = MBC.perf.DA;                         % decoding accuracy
```

```matlab
% cross-validated MBR
CV  = ML_CV(numel(x), k, 'kf');            % k-folds cross-validation
MBR = ML_MBI(Y, x, [], [], CV, 'MBR', []);
r   = MBR.perf.r;                          % predictive correlation
```

For extra input parameters of `ML_MBI.m` and fields of the output structure `MBC`/`MBR`, type `help ML_MBI` into the command window. For manipulating the prior probabilities for prediction in the test data, type `help mbitest`.


## Getting started with Python

> These tools rely on the [cvLME package](https://github.com/JoramSoch/cvLME) (`cvBMS.py`) which is currently included in the repository.

**Note: Python code has not been tested so far!** For using the Python code, you have to import the MBI module somewhere in your script:

```python
import MBI
```

Let `Y1` and `Y2` be feature matrices from training and test data and let `x1` and `x2` be natural numbers representing classes in training and test data. Then, the simplest version of MBC can be used to calculate posterior class probabilities in the test data as follows:

```python
# multivariate Bayesian classification
m1   = MBI.model(Y1, x1, mb_type='MBC')
MBA1 = m1.train()
m2   = MBI.model(Y2, x2, mb_type='MBC')
PP2  = m2.test(MBA1)
```

Let `Y1` and `Y2` be feature matrices from training and test data and let `x1` and `x2` be real numbers representing targets in training and test data. Then, the simplest version of MBR can be used to calculate posterior target densities in the test data as follows:

```python
# multivariate Bayesian regression
m1   = MBI.model(Y1, x1, mb_type='MBR')
MBA1 = m1.train()
m2   = MBI.model(Y2, x2, mb_type='MBR')
PP2  = m2.test(MBA1)
```

The output `PP2` is a matrix of posterior probabilities in the test data, with one row corresponding to one data point and each row either representing discrete probability masses or a continuous probability density.

Let `Y` be an entire feature matrix and let `x` be an entire vector of class labels or target values across all data points. Then, cross-validated MBC or MBR (number of CV folds `k`) can be easily implemented as follows:

```python
# cross-validated MBC
MBC = MBI.cvMBI(Y, x, mb_type='MBC')
MBC.crossval(k=10, cv_mode='kfc')    # k-folds on points per class
MBC.predict()
DA  = MBC.evaluate('DA')             # decoding accuracy
```

```python
# cross-validated MBR
MBR = MBI.cvMBI(Y, x, mb_type='MBR')
MBR.crossval(k=10, cv_mode='kf')     # k-folds cross-validation
MBR.predict()
r   = MBR.evaluate('r')              # predictive correlation
```

For extra input parameters of `MBI.cvMBI` and attributes of the objects `MBC`/`MBR`, see [these lines of code](https://github.com/JoramSoch/MBI/blob/main/Python/MBI.py#L140-L162). For manipulating the prior probabilities for prediction in the test data, see [these lines of code](https://github.com/JoramSoch/MBI/blob/main/Python/MBI.py#L227-L237).

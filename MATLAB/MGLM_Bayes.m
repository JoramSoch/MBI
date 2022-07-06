function [Mn, Ln, On, vn] = MGLM_Bayes(Y, X, P, M0, L0, O0, v0)
% _
% Bayesian Estimation of Multivariate GLM with Normal-Wishart Priors
% FORMAT [Mn, Ln, On, vn] = MGLM_Bayes(Y, X, P, M0, L0, O0, v0)
% 
%     Y  - an n x v data matrix of measured signals
%     X  - an n x p design matrix of predictor variables
%     P  - an n x n precision matrix specifying correlations
%     M0 - a  p x v matrix (prior means of regression coefficients)
%     L0 - a  p x p matrix (prior precision of regression coefficients)
%     O0 - a  v x v matrix (prior inverse scale matrix for covariance)
%     v0 - a  1 x 1 scalar (prior degrees of freedom for covariance)
% 
%     Mn - a  p x v matrix (posterior means of regression coefficients)
%     Ln - a  p x p matrix (posterior precision of regression coefficients)
%     On - a  v x v matrix (posterior inverse scale matrix for covariance)
%     vn - a  1 x 1 scalar (posterior degrees of freedom for covariance)
% 
% FORMAT [Mn, Ln, On, vn] = MGLM_Bayes(Y, X, P, M0, L0, O0, v0) returns
% the posterior parameter estimates for a multivariate general linear model
% with data matrix Y, design matrix X, precision matrix P and normal-
% Wishart distributed priors for regression coefficients (M0, L0) and
% signal covariance (O0, v0).
% 
% References:
% [1] Wikipedia (2021): "Bayesian multivariate linear regression";
%     URL: https://en.wikipedia.org/wiki/Bayesian_multivariate_linear_regression#Posterior_distribution.
% [2] Soch J (2020): "Posterior distribution for multivariate Bayesian linear regression";
%     URL: https://statproofbook.github.io/P/mblr-post.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 20/02/2022, 09:54


% Get model dimensions
%-------------------------------------------------------------------------%
n = size(Y,1);                  % number of observations
v = size(Y,2);                  % number of signals
p = size(X,2);                  % number of regressors

% Set precision if required
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(P)
    P = eye(n);                 % precision = identity matrix
end;

% Enlarge priors if required
%-------------------------------------------------------------------------%
if size(M0,2) == 1
    M0 = repmat(M0,[1 v]);      % make M0 a p x v matrix
end;

% Estimate posterior parameters
%-------------------------------------------------------------------------%
PY = P*Y;                       % precision matrix times data matrix
Ln = X'*P*X + L0;               % precision of regression coefficients
Mn = inv(Ln) * (X'*PY + L0*M0); % means of regression coefficients
vn = v0 + n;                    % degrees of freedom for covariance
On = O0 + Y'*PY + M0'*L0*M0 ... % inverse scale matrix for covariance
                - Mn'*Ln*Mn;
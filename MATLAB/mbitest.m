function PP = mbitest(Y, x, X, V, MBA, prior)
% _
% Testing for multivariate Bayesian inversion
% FORMAT PP = mbitest(Y, x, X, V, MBA, prior)
% 
%     Y     - an n x v matrix of feature variables
%     x     - an n x 1 vector of classes/targets
%     X     - an n x r matrix of covariate values
%     V     - an n x n matrix, the covariance between observations
%     MBA   - a structure specifying a trained MBI model (see "mbitrain")
%     prior - a structure specifying the prior model probabilities
%             o x - a 1 x L vector, the support of the prior distribution
%             o p - a 1 x L vector, the prior probability mass or density
%              (L - number of classes or number of target variable levels)
% 
%     PP    - an n x L matrix of posterior probabilities
% 
% FORMAT PP = mbitest(Y, x, X, V, MBA, prior) tests a multivariate Bayesian 
% model for classification or regression of classes or targets x from
% features Y, accounting for covariates X and covariance V.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 28/02/2025, 10:13


% Set inputs if required
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(X), X = [];            end;
if nargin < 4 || isempty(V), V = eye(numel(x)); end;

% Set prior if required
%-------------------------------------------------------------------------%
if nargin < 6 || isempty(prior)
    % discrete uniform, if classification
    if MBA.is_MBC
        prior = uniprior('disc', max(MBA.input.x));
    % continuous uniform, if regression
    else
        prior = uniprior('cont', 100, min(MBA.input.x), max(MBA.input.x));
    end;
end;    
        
% Create design matrix
%-------------------------------------------------------------------------%
Y2 = Y; clear Y;
if MBA.is_MBC
    C  = max(x);
    X2 = zeros(size(Y2,1),C);
else
    X2 = [zeros(size(x)), ones(size(x))];
end;
X2 = [X2, X];
V2 = V;

% Get data dimensions
%-------------------------------------------------------------------------%
n = size(Y2,1);
v = size(Y2,2);

% Specify prior parameters
%-------------------------------------------------------------------------%
M1 = MBA.post.M1;
L1 = MBA.post.L1;
O1 = MBA.post.O1;
v1 = MBA.post.v1;

% Calculate posterior probabilities
%-------------------------------------------------------------------------%
L     = numel(prior.x);
PP    = zeros(n,L);
logPP = zeros(n,L);
for i = 1:n                     % loop over data points in the test set
    y2i = Y2(i,:);
    x2i = X2(i,:);
    pii = 1/V2(i,i);
    for j = find(prior.p)       % loop over label values (where prior is non-zero)
        x2ij = x2i;
        if MBA.is_MBC           % classification -> categorical
            x2ij(j)   = 1;
        else                    % regression -> parametric
            x2ij(1)   = prior.x(j);
        end;                    % calculte posterior parameters
        [M2, L2, O2, v2] = MGLM_Bayes(y2i, x2ij, pii, M1, L1, O1, v1);
        logPP(i,j)       = -v/2 * logdet(L2) - v2/2 * logdet(O2) + log(prior.p(j));
      % PP(i,j)          = sqrt( (det(L2))^(-v) / (det(O2))^(v2) ) * prior.p(j);
    end
    PP(i,prior.p~=0) = exp(logPP(i,prior.p~=0)-mean(logPP(i,prior.p~=0)));
  % PP(i,:)          = exp(logPP(i,:)-mean(logPP(i,:)));
    if MBA.is_MBC               % classification -> mass
        PP(i,:) = PP(i,:)./sum(PP(i,:));
    else                        % regression -> density
        PP(i,:) = PP(i,:)./trapz(prior.x, PP(i,:));
    end;
    clear y2i x2i pii x2ij
end;


% Function: compute log-determinant
%-------------------------------------------------------------------------%
function ld = logdet(V)
% _
% Log-Determinant for Covariance Matrix
% FORMAT ld = logdet(V)
% 
%     V  - an n x n positive-definite symmetric covariance matrix
% 
%     ld - the log-determinant of that covariance matrix
% 
% FORMAT LD = logdet(V) computes the logarithm of the determinant of the
% matrix V where V is an n x n square matrix [1]. If V is singular, such
% that det(V) = 0, then it returns -Inf.
% 
% This function computes the log-determinant using Cholesky factorization,
% thus assuming that V is a (positive-definite symmetric) covariance matrix.
% 
% References:
% [1] Lin D (2008). Safe computation of logarithm-determinat of large matrix;
%     URL: http://www.mathworks.com/matlabcentral/fileexchange/22026-safe-
%     computation-of-logarithm-determinat-of-large-matrix/content/logdet.m

% compute log-determinant
ld = 2 * sum(log(diag(chol(V))));
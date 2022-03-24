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
%             o x - an 1 x L vector, the support of the prior distribution
%             o p - an 1 x L vector, the prior probability mass or density
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
% 
% First edit: 20/02/2022, 09:28
%  Last edit: 20/02/2022, 11:16


% Set inputs if required
%-------------------------------------------------------------------------%
if isempty(X) || nargin < 3, X = [];                  end;
if isempty(V) || nargin < 4, V = eye(size(numel(x))); end;

% Set prior if required
%-------------------------------------------------------------------------%
if isempty(prior) || nargin < 5
    % discrete uniform, if classification
    if MBA.is_MBC
        C = max(MBA.input.x);
        prior.x = [1:C];
        prior.p = (1/C)*ones(1,C);
    % continuous uniform, if regression
    else
        L = 100;
        prior.x = [min(MBA.input.x):(range(MBA.input.x)/(L-1)):max(MBA.input.x)];
        prior.p = (1/range(MBA.input.x))*ones(1,L);
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
L  = numel(prior.x);
PP = zeros(n,L);
logPP = zeros(n,L);
for i = 1:n                     % loop over test data points
    y2i = Y2(i,:);
    x2i = X2(i,:);
    pii = 1/V2(i,i);
    for j = 1:L                 % loop over classes/levels
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
  % PP(i,:) = exp(logPP(i,:)-mean(logPP(i,:)));
    if MBA.is_MBC               % classification -> mass
        PP(i,:) = PP(i,:)./sum(PP(i,:));
    else                        % regression -> density
        PP(i,:) = PP(i,:)./trapz(prior.x, PP(i,:));
    end;
    clear y2i x2i pii x2ij
end;
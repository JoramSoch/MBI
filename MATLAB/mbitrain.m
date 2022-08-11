function MBA = mbitrain(Y, x, X, V, type)
% _
% Training for multivariate Bayesian inversion
% FORMAT MBA = mbitrain(Y, x, X, V, type)
% 
%     Y    - an n x v matrix of feature variables
%     x    - an n x 1 vector of classes/targets
%     X    - an n x r matrix of covariate values
%     V    - an n x n matrix, the covariance between observations
%     type - a string indicating the analysis type ('MBC' or 'MBR')
% 
%     MBA  - a structure, the trained multivariate Bayesian automaton
%            o is_MBC - true, if MBI for classification
%            o input  - user input from above (except for Y)
%            o data   - data used for estimating parameters
%                       o Y1 - n x v data matrix in training
%                       o X1 - n x p design matrix in training
%                       o V1 - n x n covariance matrix in training
%            o prior  - prior parameters
%                       o M0 - p x v prior mean
%                       o L0 - p x p prior precision
%                       o O0 - v x v prior inverse scale
%                       o v0 - prior degrees of freedom
%            o post   - posterior parameters
%                       o M0 - p x v prior mean
%                       o L0 - p x p prior precision
%                       o O0 - v x v prior inverse scale
%                       o v0 - prior degrees of freedom
% 
% FORMAT MBA = mbitrain(Y, x, X, V, type) trains a multivariate Bayesian 
% model for classification or regression of classes or targets x from
% features Y, accounting for covariates X and covariance V.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 14/07/2022, 17:20


% Set inputs if required
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(X), X = [];                  end;
if nargin < 4 || isempty(V), V = eye(size(numel(x))); end;

% Set type if required
%-------------------------------------------------------------------------%
if nargin < 5 || isempty(type)
    % classification, if all are integer and at most 5 classes
    if numel(unique(x)) <= 5 && all((x-round(x))<exp(-23))
        type = 'MBC';
    % regression, otherwise
    else
        type = 'MBR';
    end;
end;    

% Create design matrix
%-------------------------------------------------------------------------%
Y1 = Y; clear Y;
if strcmp(type,'MBC')
    C  = max(x);
    X1 = zeros(size(Y1,1),C);
    for j = 1:C, X1(x==j,j) = 1; end;
elseif strcmp(type,'MBR')
    X1 = [x, ones(size(x))];
else
    warning('mbitrain:invalid_type', 'Analysis type is invalid (must be "MBC" or "MBR")!');
end;
X1 = [X1, X];
P1 = inv(V);

% Get data dimensions
%-------------------------------------------------------------------------%
n = size(Y1,1);
v = size(Y1,2);
p = size(X1,2);

% Specify prior parameters
%-------------------------------------------------------------------------%
M0 = zeros(p,v);
L0 = zeros(p,p);
O0 = zeros(v,v);
v0 = 0;

% Calculate posterior parameters
%-------------------------------------------------------------------------%
[M1, L1, O1, v1] = MGLM_Bayes(Y1, X1, P1, M0, L0, O0, v0);

% Assemble MBA structure
%-------------------------------------------------------------------------%
if strcmp(type,'MBC')
    MBA.is_MBC = true;
elseif strcmp(type,'MBR')
    MBA.is_MBC = false;
end;
MBA.input.n  = n;
MBA.input.v  = v;
MBA.input.x  = x;
MBA.input.X  = X;
MBA.input.V  = V;
MBA.data.Y1  = Y1;
MBA.data.X1  = X1;
MBA.data.V1  = V;
MBA.prior.M0 = M0;
MBA.prior.L0 = L0;
MBA.prior.O0 = O0;
MBA.prior.v0 = v0;
MBA.post.M1  = M1;
MBA.post.L1  = L1;
MBA.post.O1  = O1;
MBA.post.v1  = v1;
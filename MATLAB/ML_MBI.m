function MBI = ML_MBI(Y, x, X, V, CV, type, prior)
% _
% Cross-Validated Multivariate Bayesian Inversion
% FORMAT MBI = ML_MBI(Y, x, X, V, CV, prior, type)
% 
%     Y     - an n x v matrix of feature variables
%     x     - an n x 1 vector of classes/targets
%     X     - an n x r matrix of covariate values
%     V     - an n x n matrix, the covariance between observations
%     CV    - an n x k matrix, specifying the cross-validation folds
%     type  - a string indicating the analysis type ('MBC' or 'MBR')
%     prior - a structure specifying the prior distribution (see "mbitest")
% 
%     MBI   - a structure specifying the performed MBI (to be documented)
% 
% FORMAT MBI = ML_MBI(Y, x, X, V, CV, prior, type) splits measured signals
% Y, classes/targets x and covariate values X into cross-validation folds
% according to CV and then performs multivariate Bayesian inversion for
% classification into categories or regression on targets.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% 
% First edit: 20/02/2022, 10:58
%  Last edit: 20/02/2022, 10:58


% Set inputs if required
%-------------------------------------------------------------------------%
if isempty(X)  || nargin < 3, X  = [];                  end;
if isempty(V)  || nargin < 4, V  = eye(size(numel(x))); end;
if isempty(CV) || nargin < 5, CV = ML_CV(x, 10, 'kfc'); end;

% Set type if required
%-------------------------------------------------------------------------%
if isempty(type) || nargin < 6
    % classification, if all are integer and at most 5 classes
    if numel(unique(x)) <= 5 && all((x-round(x))<exp(-23))
        type = 'MBC';
    % regression, otherwise
    else
        type = 'MBR';
    end;
end;

% Set prior if required
%-------------------------------------------------------------------------%
if isempty(prior) || nargin < 7
    % discrete uniform, if classification
    if strcmp(type,'MBC')
        C = max(x);
        prior.x = [1:C];
        prior.p = (1/C)*ones(1,C);
    % continuous uniform, if regression
    elseif strcmp(type,'MBR')
        L = 100;
        prior.x = [min(x):(range(x)/(L-1)):max(x)];
        prior.p = (1/range(x))*ones(1,L);
    else
        warning('ML_MBI:invalid_type', 'Analysis type is invalid (must be "MBC" or "MBR")!');
    end;
end;

% Get data dimensions
%-------------------------------------------------------------------------%
n = size(Y,1);
v = size(Y,2);
k = size(CV,2);

% Get number of classes
%-------------------------------------------------------------------------%
if strcmp(type,'MBC')
    C = max(x);
    N = sum(repmat(x,[1 C])==repmat([1:C],[n 1]), 1);
elseif strcmp(type,'MBR')
    C = 1;
    N = n;
else
    warning('ML_MBI:invalid_type', 'Analysis type is invalid (must be "MBC" or "MBR")!');
end;
p = strcmp(type,'MBC')*C + strcmp(type,'MBR')*2 + size(X,2);

% Prepare analysis display
%-------------------------------------------------------------------------%
fprintf('\n');
fprintf('-> Multivariate Bayesian inversion:\n');
fprintf('   - %d x %d design matrix;\n', n, p);
fprintf('   - %d x %d data matrix;\n', n, v);
fprintf('   - k = %d CV folds;\n', k);
if strcmp(type,'MBC')
    fprintf('   - C = %d classes;\n', C);
elseif strcmp(type,'MBR')
    fprintf('   - 1 target variable;\n');
end;
fprintf('\n');
fprintf('-> Cross-validated estimation:\n');

% Cross-validated inversion
%-------------------------------------------------------------------------%
L  = numel(prior.x);            % classes/levels
xt = zeros(n,1);                % "true" classes
xp = zeros(n,1);                % predicted classes
PP = zeros(n,L);                % posterior probabilities
for g = 1:k
    
    fprintf('   - CV fold %d: ', g);
    % get test and training set
    i1 = find(CV(:,g)==1);      % indices
    i2 = find(CV(:,g)==2);
    Y1 = Y(i1,:);               % data
    Y2 = Y(i2,:);
    x1 = x(i1);                 % classes/targets
    x2 = x(i2);
    if ~isempty(X)              % covariates
        X1 = X(i1,:);               
        X2 = X(i2,:);
    else
        X1 = [];
        X2 = [];
    end;
    V1 = V(i1,i1);              % covariance
    V2 = V(i2,i2);
    
    fprintf('training, ');
    % training data: X is known, infer on B/T
    MBA1 = mbitrain(Y1, x1, X1, V1, type);
    
    fprintf('test, ');
    % test data: B/T are known, infer on X
    PP(i2,:) = mbitest(Y2, x2, X2, V2, MBA1, prior);
    
    fprintf('done.\n');
    % collect true and predicted
    for i = 1:numel(i2)
        xt(i2(i)) = x2(i);
        xp(i2(i)) = prior.x(PP(i2(i),:)==max(PP(i2(i),:)));
    end;

end;
clear i1 i2 Y1 Y2 x1 x2 X1 X2 V1 V2
fprintf('\n');

% Calculate performance (MBC)
%-------------------------------------------------------------------------%
if strcmp(type,'MBC')
    DA = mean(xp==xt);      % decoding accuracy
    CA = zeros(C,1);
    for j = 1:C             % class accuracies
        CA(j,1) = mean(xp(xt==j)==j);
    end;                    % balanced accuracy
    BA = mean(CA);
    [ph, DA_CI] = binofit(uint16(round(DA*n)),   n,  0.1);
    [ph, BA_CI] = binofit(uint16(floor(BA*n)),   n,  0.1);
    [ph, CA_CI] = binofit(uint16(round(CA.*N')), N', 0.1);
    CM = zeros(C,C);
    for j = 1:C             % confusion matrix
        CM(:,j) = mean(repmat(xp(xt==j),[1 C])==repmat([1:C],[sum(xt==j) 1]))';
    end;
end;

% Calculate performance (MBR)
%-------------------------------------------------------------------------%
[a, b, c, d] = corrcoef(xp, xt, 'Alpha', 0.1);
r   =  a(1,2);                  % correlation coefficient
r_p =  b(1,2);                  % correlation p-value
r_CI= [c(1,2), d(1,2)];         % 90% confidence interval
mn  = polyfit(xt,xp,1);         % slope and intercept
R2  = r.^2;
MAE = mean(abs(xp-xt));
MSE = mean((xp-xt).^2);
clear a b c d

% Assemble MBI structure
%-------------------------------------------------------------------------%
if strcmp(type,'MBC')
    MBI.is_MBC = true;
elseif strcmp(type,'MBR')
    MBI.is_MBC = false;
end;
MBI.data.Y     = Y;
MBI.data.x     = x;
MBI.data.X     = X;
MBI.data.V     = V;
MBI.data.C     = C;
MBI.data.N     = N;
MBI.pars.CV    = CV;
MBI.pars.prior = prior;
MBI.pred.PP    = PP;
MBI.pred.xt    = xt;
MBI.pred.xp    = xp;
if strcmp(type,'MBC')
    MBI.perf.DA    = DA;
    MBI.perf.BA    = BA;
    MBI.perf.CA    = CA;
    MBI.perf.DA_CI = DA_CI;
    MBI.perf.BA_CI = BA_CI;
    MBI.perf.CA_CI = CA_CI;
    MBI.perf.CM    = CM;
elseif strcmp(type,'MBR')
    MBI.perf.r     = r;
    MBI.perf.r_p   = r_p;
    MBI.perf.r_CI  = r_CI;
    MBI.perf.R2    = R2;
    MBI.perf.MAE   = MAE;
    MBI.perf.MSE   = MSE;
    MBI.perf.mn    = mn;
end;
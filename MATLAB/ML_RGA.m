function RGA = ML_RGA(x, xs, mode)
% _
% Rank Graduation Accuracy as Unified Measure of Predictive Performance
% FORMAT RGA = ML_RGA(x, xs, mode)
% 
%     x    - an n x 1 vector of binary class labels (0, 1) or
%            an n x 1 vector of multiple class labels (1, 2, 3 etc.) or
%            an n x 1 vector of regression target values (real-valued)
%     xs   - an n x 1 vector of predictor scores (decision values) or
%            an n x C matrix of predictor scores (one-vs-rest decisions)
%     mode - a string indicating the averaging method for n-ary
%            classification ('RGA', 'macro' or 'weighted')
% 
%     RGA  - a scalar, the rank graduation accuracy (0 <= RGA <= 1)
% 
% FORMAT RGA = ML_RGA(x, xs, mode) computes the rank graduation accuracy
% [1] for the task of predicting labels x using decision values xs (e.g.
% SVM score values) and applying the averaging method mode, if the task
% is one of multi-class classification.
% 
% The predictors scores xs can be, for example,
% - Bayesian posterior probabilities p(x_i=j|y_i)
% - logistic regression linear predictors
% - random forest class probabilities
% - SVM decision function values
% - gradient boosting scores, etc.
% 
% References:
% [1] Giudici P, Raffinetti E (2025): "RGA: a unified measure of predictive
%     accuracy". Advances in Data Analysis and Classification, vol. 19,
%     iss. 1, pp. 67-93.
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% Edited: 12/03/2026, 11:32


% Set defaults values
%-------------------------------------------------------------------------%
if nargin < 3 || isempty(mode)
    if size(xs,2) == 1
        mode = 'RGA';
    else
        mode = 'weighted';
    end;
end;

% Get data dimensions
%-------------------------------------------------------------------------%
n = numel(x);                   % number of data points
C = size(xs,2);                 % number of classes
i = [1:n]';                     % data point indices

% Calculate RGA measure
%-------------------------------------------------------------------------%
if C == 1                       % regression or binary classification
    % obtain ranks
    [xs_sort, r ] = sort(x ,1,'ascend');
    [xs_sort, rs] = sort(xs,1,'ascend');
    clear xs_sort
    % compute RGA
    RGA = (sum(i.*x(rs)) - sum(i.*x(r(n+1-i)))) / ...
          (sum(i.*x(r )) - sum(i.*x(r(n+1-i))));
else                            % multi-class classification
    % compute class-wise RGAs
    RGA = zeros(1,C);
    w   = zeros(1,C);
    for j = 1:C
        xj     = 1*(x==j) + 0*(x~=j);
        xsj    = xs(:,j);
        RGA(j) = ML_RGA(xj, xsj, 'RGA');
        w(j)   = sum(xj)/n;
    end;
    % average class-wise RGAs
    if strcmp(mode,'macro')
        RGA = mean(RGA);
    elseif strcmp(mode,'weighted')
        RGA = sum(w.*RGA);
    end;
end;
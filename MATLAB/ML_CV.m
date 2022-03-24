function CV = ML_CV(c, k, mode)
% _
% Cross-Validation Folds for Machine Learning Analysis
% FORMAT CV = ML_CV(c, k, mode)
% 
%     c    - an n x 1 vector of class labels (1, 2, 3 etc.)
%     k    - an integer larger than 1, the number of CV folds
%     mode - a string indicating the cross-validation mode
%            o 'kf'   - k-folds cross-validation across all points
%            o 'kfc'  - k-folds cross-validation on points per class
%            o 'loo'  - leave-one-out cross-validation across all points
%            o 'looc' - leave-one-out cross-validation on points per class
% 
%     CV   - an n x k matrix indicating training (1) and test (2)
%            data for each cross-validation fold
% 
% FORMAT CV = ML_CV(c, k, mode) splits n observations from classes c into
% k cross-validation folds according to cross-validation strategy mode.
% 
% Notes:
% -  if c is a scalar, this is taken to be n;
% -  the default value for k is 10;
% -  the default value of mode is 'kfc';
% - 'loo' automatically sets k to n;
% - 'looc' requires that all classes are equally large;
%    if this is not fulfilled, mode is set to 'loo'.
% 
% Author: Joram Soch, DZNE Göttingen
% E-Mail: Joram.Soch@DZNE.de
% 
% First edit: 06/07/2021, 13:14
%  Last edit: 02/08/2021, 15:02


% Set defaults values
%-------------------------------------------------------------------------%
if numel(c) == 1,               c = ones(c,1); end;
if nargin < 2 || isempty(k),    k = 10;        end;
if nargin < 3 || isempty(mode), mode = 'kfc';  end;
if strcmp(mode,'loo'),          k = numel(c);  end;

% Get class indices
%-------------------------------------------------------------------------%
n  = numel(c);
C  = max(c);
ic = cell(1,C);
nc = zeros(1,C);
for j = 1:C
    ic{j} = find(c==j);
    nc(j) = numel(ic{j});
end;

% Create CV folds
%-------------------------------------------------------------------------%
if k < 2                        % less than 2 CV folds
    k = 2;
    warning('ML_CV:few_folds', 'Number of CV folds is too small. Number of CV folds increased to k = %d!', k);
end;
if strcmp(mode,'looc')
    if numel(unique(nc)) > 1    % unequal class sizes
        mode = 'loo';
        k = numel(c);
        warning('ML_CV:unequal_classes', 'Classes have different sizes. CV mode set to mode = "%s"!', mode);
    else                        % equal class sizes
        k = nc(1);
    end;
end;
CV = zeros(n,k);
% k-folds and leave-one-out cross-validation
if strcmp(mode,'kf') || strcmp(mode,'loo')
    nf = ceil(n/k);
    is = [1:n]';
    for g = 1:k
        i2 = is([((g-1)*nf+1):min([g*nf, n])]);
        i1 = setdiff(is,i2);
        CV(i1,g) = 1;           % training data points
        CV(i2,g) = 2;           % test data points
    end;
end;
% cross-validation on points per class
if strcmp(mode,'kfc') || strcmp(mode,'looc')
    nf = ceil(nc./k);
    is = [1:n]';
    for g = 1:k
        i2 = [];
        for j = 1:C
            i2 = [i2; ic{j}([((g-1)*nf(j)+1):min([g*nf(j), nc(j)])])];
        end;
        i1 = setdiff(is,i2);
        CV(i1,g) = 1;           % training data points
        CV(i2,g) = 2;           % test data points
    end;
end;
% remove folds, if necessary
if ~isempty(find(var(CV)==0))
    CV = CV(:,var(CV)>0);
    warning('ML_CV:empty_folds', 'Number of points (per class) and number of CV folds results in empty CV fold. Number of CV folds reduced to k = %d!', size(CV,2));
end;
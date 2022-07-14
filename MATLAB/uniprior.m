function prior = uniprior(type, L, x_min, x_max)
% _
% Mass or Density Function for a Uniform Prior Distribution
% FORMAT prior = uniprior(type, L, x_min, x_max)
%     type  - a string indicating the type of random variable
%             o 'disc' - discrete random variable
%             o 'cont' - continuous random variable
%     L     - an integer, the number of possible values
%     x_min - a scalar, the minimum possible value (only if type is 'cont')
%     x_max - a scalar, the maximum possible value (only if type is 'cont')
%     
%     prior - a structure specifying the prior model probabilities
%             o x - a 1 x L vector, the support of the prior distribution
%             o p - a 1 x L vector, the prior probability mass or density
% 
% FORMAT prior = uniprior(type, L, x_min, x_max) generates a discrete or 
% continuous prior distribution using L levels, i.e. number of classes L,
% if type is 'disc' (discrete random variable) or L possible values between
% and including x_min and x_max, if type is 'cont'.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% Edited: 14/07/2022, 17:18


% Set default values, if necessary
%-------------------------------------------------------------------------%
if nargin < 1 || isempty(type), type = 'disc'; end;
if nargin < 2 || isempty(L),    L    = 100;    end;

% Discrete random variable (classes)
%-------------------------------------------------------------------------%
if strcmp(type,'disc')
    C = L;
    prior.x = [1:C];
    prior.p = (1/C)*ones(1,C);
end;
    
% Continuous random variable (targets)
%-------------------------------------------------------------------------%
if strcmp(type,'cont')
    prior.x = [x_min:((x_max-x_min)/(L-1)):x_max];
    prior.p = (1/(x_max-x_min)) * ones(1,L);
end;
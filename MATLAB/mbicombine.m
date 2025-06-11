function MBA = mbicombine(MBAs, prior)
% _
% Combination of multivariate Bayesian automata
% FORMAT MBA = mbicombine(MBAs, prior)
% 
%     MBAs  - a 1 x S structure, the trained
%             multivariate Bayesian automata (see "mbitrain")
%             o is_MBC - true, if MBI for classification
%             o prior  - prior parameters
%                        o M0 - p x v prior mean
%                        o L0 - p x p prior precision
%                        o O0 - v x v prior inverse scale
%                        o v0 - prior degrees of freedom
%             o post   - posterior parameters
%                        o M0 - p x v prior mean
%                        o L0 - p x p prior precision
%                        o O0 - v x v prior inverse scale
%                        o v0 - prior degrees of freedom
%     prior - an integer, indexing from which element of MBAs
%             to use prior information (0 -> non-informative prior)
% 
%     MBA   - a structure, the combined
%             multivariate Bayesian automaton (suitable for "mbitest")
%             o is_MBC - true, if MBI for classification
%             o prior  - prior parameters (see above)
%             o post   - posterior parameters (see above)
% 
% FORMAT MBA = mbicombine(MBAs, prior) combines posterior distributions
% from multivariate Bayesian automata MBAs, using the prior distribution
% from MBAs(prior), and returns the combined multivariate Bayesian
% automaton MBA according to the principle of multivariate Bayesian
% swarm learning.
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% Edited: 15/04/2025, 14:47


% Set inputs if required
%-------------------------------------------------------------------------%
if nargin < 2 || isempty(prior), prior = 0; end;

% Get model dimensions
%-------------------------------------------------------------------------%
p = size(MBAs(1).post.M1,1);
v = size(MBAs(1).post.M1,2);
S = numel(MBAs);

% Specify prior distribution
%-------------------------------------------------------------------------%
MBA.is_MBC = MBAs(1).is_MBC;
if prior == 0
    MBA.prior.M0 = zeros(p,v);
    MBA.prior.L0 = zeros(p,p);
    MBA.prior.O0 = zeros(v,v);
    MBA.prior.v0 = 0;
else
    MBA.prior.M0 = MBAs(prior).prior.M0;
    MBA.prior.L0 = MBAs(prior).prior.L0;
    MBA.prior.O0 = MBAs(prior).prior.O0;
    MBA.prior.v0 = MBAs(prior).prior.v0;
end;

% Calculate posterior distribution
%-------------------------------------------------------------------------%
MBA.post.M1 = MBA.prior.M0;
MBA.post.L1 = MBA.prior.L0;
MBA.post.O1 = MBA.prior.O0;
MBA.post.v1 = MBA.prior.v0;
for k = 1:S
    MBA.post.M1 = MBA.post.M1 + (MBAs(k).post.L1 * MBAs(k).post.M1 - MBAs(k).prior.L0 * MBAs(k).prior.M0);
    MBA.post.L1 = MBA.post.L1 + (MBAs(k).post.L1 - MBAs(k).prior.L0);
    MBA.post.O1 = MBA.post.O1 + (MBAs(k).post.O1 - MBAs(k).prior.O0- ...
                                 MBAs(k).prior.M0'*MBAs(k).prior.L0* MBAs(k).prior.M0 + ...
                                 MBAs(k).post.M1'* MBAs(k).post.L1 * MBAs(k).post.M1);
    MBA.post.v1 = MBA.post.v1 + (MBAs(k).post.v1 - MBAs(k).prior.v0);
end;
MBA.post.M1 = inv(MBA.post.L1)*  MBA.post.M1;
MBA.post.O1 =     MBA.post.O1 -  MBA.post.M1'* MBA.post.L1 * MBA.post.M1;
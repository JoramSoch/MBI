function R = matnrnd(M,U,V,c,A,B)
% _
% Random Matrices from the Matrix-Variate Normal Distribution
% FORMAT R = MD_matnrnd(M,U,V,c,A,B)
% 
%     M - an n x p matrix, the mean of the matrix normal distribution
%     U - an n x n matrix, the covariance across rows of the matrix
%     V - a  p x p matrix, the covariance across columns of the matrix
%     c - an integer, the number of cases to be drawn from the distribution
%     A - an n x n matrix, the upper Cholesky decomposition of U
%     B - a  p x p matrix, the lower Cholesky decomposition of V
% 
%     R - an n x p x c array of random matrices from the distribution
% 
% FORMAT R = MD_matnrnd(M,U,V,c,A,B) draws c random values from the matrix
% normal distribution with mean M, row covariance U, column covariance V.
% This is done by sampling R from the distribution X ~ MN(0_np, I_n, I_p)
% and transforming it into Y = M + A*X*B ~ MN(M, A*A', B'*B) where A and B
% are obtained by Cholesky decomposition of U and V [1].
% 
% References:
% [1] Wikipedia: "Matrix normal distribution";
%     URL: https://en.wikipedia.org/wiki/Matrix_normal_distribution.
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% 
% First edit: 07/03/2018, 08:55
%  Last edit: 06/03/2020, 15:23


% Set number of cases if necessary
%-------------------------------------------------------------------------%
if nargin < 4 || isempty(c), c = 1; end;

% Compute cholesky factors of U and V
%-------------------------------------------------------------------------%
if nargin < 5 || isempty(A), A = chol(U)'; end;
if nargin < 6 || isempty(B), B = chol(V);  end;

% Sample from standard normal distribution
%-------------------------------------------------------------------------%
R = randn([size(U,1), size(V,1), c]);
%   normrnd(0,1,[size(U,1), size(V,1), c]);

% Transform into matrix normal distribution
%-------------------------------------------------------------------------%
for i = 1:c
    R(:,:,i) = M + A*R(:,:,i)*B;
end;
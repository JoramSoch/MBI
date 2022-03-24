function ld = logdet(V)
% _
% Log-Determinant for Multivariate Normal Covariance Matrix
% FORMAT ld = logdet(V)
% 
%     V  - an n x n multivariate normal covariance matrix
% 
%     ld - the log-determinant of that covariance matrix
% 
% FORMAT LD = MD_mvn_logdet(V) computes the logarithm of the determinant
% of the matrix V where V is an n x n square matrix [1]. If V is singular,
% such that det(V) = 0, then it returns -Inf.
% 
% This function computes the log-determinant using Cholesky factorization,
% thereby assuming that V is a (positive definite) covariance matrix.
% 
% References:
% [1] http://www.mathworks.com/matlabcentral/fileexchange/22026-safe-
%     computation-of-logarithm-determinat-of-large-matrix/content/logdet.m
% 
% Author: Joram Soch, BCCN Berlin
% E-Mail: joram.soch@bccn-berlin.de
% 
% First edit: 12/02/2015, 01:30
%  Last edit: 21/05/2019, 12:45


% Compute log-determinant
%-------------------------------------------------------------------------%
ld = 2 * sum(log(diag(chol(V))));
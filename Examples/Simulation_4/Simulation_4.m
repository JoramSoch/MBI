% Multivariate Bayesian Inversion for Classification and Regression
% Simulation 4: continuous prediction (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/02/2022, 22:38: first version
% - 20/02/2025, 16:48: aligned with Python
% - 30/01/2026, 17:44: recorded analysis time


clear
close all

%%% Step 1: specify ground truth %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set ground truth
rng(2);
n  = 200;                       % number of data points
k  = 10;                        % number of CV folds
v  = 10;                        % number of features
mu = 0;                         % beta mean
sb = 1;                         % beta variance
s2 = 1;                         % noise variance
tau= 0.25;                      % time constant
V  = toeplitz(tau.^[0:1:(n-1)]);% temporal covariance
ny = 0.5;                       % space constant
Si = toeplitz( ny.^[0:1:(v-1)]);% spatial covariance

% generate targets
xm = 1;                         % -1 < x < +1
x  = (2*xm)*rand(n,1)-xm;       % x ~ U(-1,+1)
X  = [x, ones(size(x))];        % design matrix

% histogram of targets
lim = 1.5;
dx  = 0.1;
xb  = [(-lim+dx/2):dx:(+lim-dx/2)];
nb  = hist(x, xb);


%%% Step 2: generate & analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate data
B = normrnd(mu, sqrt(sb), [size(X,2) v]);
E = matnrnd(zeros(n,v), s2*V, Si, 1);
Y = X*B + E;

% specify cross-validation
CV = ML_CV(ones(size(x)), k, 'kf');

% Analysis 1: MBC
tic;
prior.x = [-xm:0.01:+xm];
prior.p = (1/range(prior.x))*ones(size(prior.x));
MBR     = ML_MBI(Y, x, [], V, CV, 'MBR', prior);
tA      = toc;

% Analysis 2: SVM
tic;
SVR = ML_SVR(x, Y, CV, 1, 1);
tB  = toc;

% store analysis time
time = {'Simulation 4', 'Figure 6B/C', tA, tB}; 
save('Simulation_4.mat', 'time');


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% prepare plotting
x_MBR  = xm;
x_SVR  = max(abs(SVR.pred.xp));
nb_MBR = hist(MBR.pred.xp, xb);
nb_SVR = hist(SVR.pred.xp, xb);


% open figure
figure('Name', 'Simulation 4', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

% plot labels
subplot(2,3,1);
bar(xb, nb, 'FaceColor', [3/4, 3/4, 3/4]);
axis([-lim, +lim, 0, (11/10)*max(nb)]);
xlabel('target value', 'FontSize', 12);
ylabel('number of samples', 'FontSize', 12);
title('Training Data', 'FontSize', 16);
text(0, -(2.5/10)*max(nb), 'MBR: posterior distributions', 'FontWeight', 'Bold', 'FontSize', 16, ...
     'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');

 % plot MBR predictions
subplot(2,3,2); hold on;
plot([-x_SVR, +x_SVR], [-x_SVR, +x_SVR], '-k', 'LineWidth', 1);
plot(x, MBR.pred.xp, '.b', 'MarkerSize', 10);
axis([-x_SVR, +x_SVR, -x_SVR, +x_SVR]);
axis square;
set(gca,'Box','On');
xlabel('actual target values', 'FontSize', 12);
ylabel('predicted target values', 'FontSize', 12);
title('MBR: MAP estimates', 'FontSize', 16);
text(-x_MBR, +x_MBR, sprintf('r = %0.2f, MAE = %0.2f', MBR.perf.r, MBR.perf.MAE), ...
     'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Bottom');

 % plot SVR predictions
subplot(2,3,3); hold on;
plot([-x_SVR, +x_SVR], [-x_SVR, +x_SVR], '-k', 'LineWidth', 1);
plot(x, SVR.pred.xp, '.r', 'MarkerSize', 10);
axis([-x_SVR, +x_SVR, -x_SVR, +x_SVR]);
axis square;
set(gca,'Box','On');
xlabel('actual target values', 'FontSize', 12);
ylabel('predicted target values', 'FontSize', 12);
title('SVR: SVM predictions', 'FontSize', 16);
text(-x_MBR, +x_MBR, sprintf('r = %0.2f, MAE = %0.2f', SVR.perf.r, SVR.perf.MAE), ...
     'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Bottom');

 % plot MBR histogram
subplot(2,3,5);
bar(xb, nb_MBR, 'b');
axis([-lim, +lim, 0, (11/10)*max(nb_MBR)]);
axis square;
xlabel('target value', 'FontSize', 12);
ylabel('number of predictions', 'FontSize', 12);
title('MBR: prediction distribution', 'FontSize', 16);

% plot SVR histogram
subplot(2,3,6);
bar(xb, nb_SVR, 'r');
axis([-lim, +lim, 0, (11/10)*max(nb_SVR)]);
axis square;
xlabel('target value', 'FontSize', 12);
ylabel('number of predictions', 'FontSize', 12);
title('SVR: prediction distribution', 'FontSize', 16);

% plot MBR posteriors
for h = 1:4
    subplot(4,6,(h>0)*12+(h>2)*4+h); hold on;
    plot(MBR.pred.xt(h), (1/10)*max(MBR.pred.PP(h,:)), 'xk', 'MarkerSize', 7.5, 'LineWidth', 2);
    plot(MBR.pred.xp(h), (1/10)*max(MBR.pred.PP(h,:)), '.b', 'MarkerSize', 20);
    plot(prior.x, MBR.pred.PP(h,:), '-b', 'LineWidth', 1);
    axis([-x_MBR, +x_MBR, 0, (11/10)*max(MBR.pred.PP(h,:))]);
    set(gca,'Box','On');
    if h == 4
        legend('true', 'mode', 'Location', 'NorthEast');
    end;
    if h == 3
        xlabel('target value', 'FontSize', 12);
        ylabel('posterior density', 'FontSize', 12);
    end;
end;
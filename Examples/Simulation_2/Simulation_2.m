% Multivariate Bayesian Inversion for Classification and Regression
% Simulation 2: two classes with confound (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/02/2022, 15:02: first version
% - 30/05/2022, 05:14: minor changes
% - 17/02/2025, 15:25: added decision boundaries
% - 19/02/2025, 11:49: aligned with Python
% - 30/01/2026, 17:32: recorded analysis time


clear
close all

%%% Step 1: specify ground truth %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set ground truth
rng(1);
mu = 1;                         % class means
b3 = 2;                         % covariate effect
Si = [1, 0.5; 0.5, 1];          % feature covariance
s2 = 4;                         % noise variance
n  = 250;                       % number of data points
k  = 10;                        % number of CV folds
v  = 2;                         % number of features
C  = 2;                         % number of classes

% generate classes
x = [kron([1:C]',ones(n/C,1)), rand(n,1)];
x = sortrows(x,2);
x  = x(:,1);                    % randomized labels
X  = zeros(n,C);                % design matrix
V  = eye(n);                    % observation covariance
for i = 1:n
    X(i,x(i)) = 1;
end;

% generate covariate
c = 1.5*rand(n,1)-0.75;         % c ~ U(-0.75, +0.75)
c(x==1) = c(x==1) + 0.25;       % x=1 -> -1 < c < 0.5
c(x==2) = c(x==2) - 0.25;       % x=2 -> -0.5 < c < 1
X = [X, c];                     % complete design matrix


%%% Step 2: generate & analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate data
B = [-mu, +mu;
     +mu, -mu;
       0,  b3];
E = matnrnd(zeros(n,v), s2*V, Si, 1);
Y = X*B + E;

% specify cross-validation
CV = ML_CV(x, k, 'kfc');

% prepare decision boundary
lim = 6;
dxy = 0.05;
Y2a = [-lim*ones((2*lim)/dxy+1,1), [-lim:dxy:+lim]'];
Y2b = [+lim*ones((2*lim)/dxy+1,1), [-lim:dxy:+lim]'];
x2  = [1, 2*ones(1,size(Y2a,1)-1)]';

% Analysis 1: MBC w/o covariate
tic;
MBC(1) = ML_MBI(Y, x, [], V, CV, 'MBC', []);
tA1    = toc;

% Analysis 2: MBC with covariate
tic;
MBC(2) = ML_MBI(Y, x, c, V, CV, 'MBC', []);
tA2    = toc;

% Analysis 3: SVM w/o covariate
tic;
SVC(1) = ML_SVC(x, Y, CV, 1, 1, 0);
tB1    = toc;

% Analysis 4: SVM with prior regression
tic;
Xc     = [c, ones(size(c))];
Yr     = (eye(n) - Xc*(Xc'*Xc)^(-1)*Xc')*Y;
SVC(2) = ML_SVC(x, Yr, CV, 1, 1, 0);
tB2    = toc;

% Analysis 1: decision boundary
MBA = mbitrain(Y, x, [], V, 'MBC');
PPa = mbitest(Y2a, x2, [], V, MBA);
PPb = mbitest(Y2b, x2, [], V, MBA);
[PP_min, ka] = min(abs(PPa(:,1)-PPa(:,2)));
[PP_min, kb] = min(abs(PPb(:,1)-PPb(:,2)));
Y_MBC        = [Y2a(ka,:); Y2b(kb,:)];
clear PP_min ka kb

% Analysis 3: decision boundary
SVM = svmtrain(x, Y, '-s 0 -t 0 -c 1 -q');
xpa = svmpredict(x2, Y2a, SVM, '-q');
xpb = svmpredict(x2, Y2b, SVM, '-q');
[x_diff, ka] = max(abs(diff(xpa)));
[x_diff, kb] = max(abs(diff(xpb)));
Y_SVC        = [mean(Y2a(ka:(ka+1),:)); mean(Y2b(kb:(kb+1),:))];
clear xp_diff ka kb

% store analysis time
time = {'Simulation 2', 'Figure 4C/E', tA1, tB1;
        'Simulation 2', 'Figure 4D/F', tA2, tB2}; 
save('Simulation_2.mat', 'time');


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create colormap
cmap = colormap;
nc   = size(cmap,1);

% open figure
figure('Name', 'Simulation 2', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

% plot features
subplot(2,3,1); hold on;
plot(Y(x==1,1), Y(x==1,2), '.r', 'MarkerSize', 10);
plot(Y(x==2,1), Y(x==2,2), 'sb', 'MarkerSize', 1, 'LineWidth', 2);
plot([-mu, +mu], [+mu, -mu], 'xk', 'MarkerSize', 12, 'LineWidth', 2);
axis([-lim, +lim, -lim, +lim]);
axis square;
set(gca,'Box','On');
legend('class 1', 'class 2', 'Location', 'NorthEast');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('Classes', 'FontSize', 16);

% plot confound
subplot(2,3,4); hold on;
for i = 1:n
    plot(Y(i,1), Y(i,2), '.', 'MarkerSize', 10, 'Color', cmap(1+round(((c(i)-min(c))/range(c))*(nc-1)),:));
end;
plot([0, 0, -1/8*b3, 0, +1/8*b3], [0, b3, 6/8*b3, b3, 6/8*b3], '-k', 'LineWidth', 2);
axis([-lim, +lim, -lim, +lim]);
axis square;
caxis([-1, +1]);
cb = colorbar;
cb.Label.String = 'covariate value';
set(gca,'Box','On');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('Confound', 'FontSize', 16);

% plot MBC w/o correction
subplot(2,3,2); hold on;
plot(Y(x==1 & MBC(1).pred.xp==1,1), Y(x==1 & MBC(1).pred.xp==1,2), '.r', 'MarkerSize', 10);
plot(Y(x==2 & MBC(1).pred.xp==1,1), Y(x==2 & MBC(1).pred.xp==1,2), 'sr', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y(x==1 & MBC(1).pred.xp==2,1), Y(x==1 & MBC(1).pred.xp==2,2), '.b', 'MarkerSize', 10);
plot(Y(x==2 & MBC(1).pred.xp==2,1), Y(x==2 & MBC(1).pred.xp==2,2), 'sb', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y_MBC(:,1), Y_MBC(:,2), '-k', 'Color', 0.5*[1,1,1], 'LineWidth', 1);
axis([-lim, +lim, -lim, +lim]);
axis square;
set(gca,'Box','On');
legend('class 1, predicted 1', 'class 2, predicted 1', 'class 1, predicted 2', 'class 2, predicted 2', 'Location', 'NorthWest');
text(+(9/10)*lim, -(9/10)*lim, sprintf('CA = %2.2f %%', MBC(1).perf.CA*100), ...
     'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('MBC w/o correction', 'FontSize', 16);

% plot MBC with covariate inclusion
subplot(2,3,5); hold on;
plot(Y(x==1 & MBC(2).pred.xp==1,1), Y(x==1 & MBC(2).pred.xp==1,2), '.r', 'MarkerSize', 10);
plot(Y(x==2 & MBC(2).pred.xp==1,1), Y(x==2 & MBC(2).pred.xp==1,2), 'sr', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y(x==1 & MBC(2).pred.xp==2,1), Y(x==1 & MBC(2).pred.xp==2,2), '.b', 'MarkerSize', 10);
plot(Y(x==2 & MBC(2).pred.xp==2,1), Y(x==2 & MBC(2).pred.xp==2,2), 'sb', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y_MBC(:,1), Y_MBC(:,2), '-k', 'Color', 0.5*[1,1,1], 'LineWidth', 1);
axis([-lim, +lim, -lim, +lim]);
axis square;
set(gca,'Box','On');
text(+(9/10)*lim, -(9/10)*lim, sprintf('CA = %2.2f %%', MBC(2).perf.CA*100), ...
     'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('MBC with covariate inclusion', 'FontSize', 16);

% plot SVC w/o correction
subplot(2,3,3); hold on;
plot(Y(x==1 & SVC(1).pred.xp==1,1), Y(x==1 & SVC(1).pred.xp==1,2), '.r', 'MarkerSize', 10);
plot(Y(x==2 & SVC(1).pred.xp==1,1), Y(x==2 & SVC(1).pred.xp==1,2), 'sr', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y(x==1 & SVC(1).pred.xp==2,1), Y(x==1 & SVC(1).pred.xp==2,2), '.b', 'MarkerSize', 10);
plot(Y(x==2 & SVC(1).pred.xp==2,1), Y(x==2 & SVC(1).pred.xp==2,2), 'sb', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y_SVC(:,1), Y_SVC(:,2), '-k', 'Color', 0.5*[1,1,1], 'LineWidth', 1);
axis([-lim, +lim, -lim, +lim]);
axis square;
set(gca,'Box','On');
text(+(9/10)*lim, -(9/10)*lim, sprintf('CA = %2.2f %%', SVC(1).perf.DA*100), ...
     'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('SVC w/o correction', 'FontSize', 16);

% plot MBC with prior regression
subplot(2,3,6); hold on;
plot(Y(x==1 & SVC(2).pred.xp==1,1), Y(x==1 & SVC(2).pred.xp==1,2), '.r', 'MarkerSize', 10);
plot(Y(x==2 & SVC(2).pred.xp==1,1), Y(x==2 & SVC(2).pred.xp==1,2), 'sr', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y(x==1 & SVC(2).pred.xp==2,1), Y(x==1 & SVC(2).pred.xp==2,2), '.b', 'MarkerSize', 10);
plot(Y(x==2 & SVC(2).pred.xp==2,1), Y(x==2 & SVC(2).pred.xp==2,2), 'sb', 'MarkerSize', 1, 'LineWidth', 2);
plot(Y_SVC(:,1), Y_SVC(:,2), '-k', 'Color', 0.5*[1,1,1], 'LineWidth', 1);
axis([-lim, +lim, -lim, +lim]);
axis square;
set(gca,'Box','On');
text(+(9/10)*lim, -(9/10)*lim, sprintf('CA = %2.2f %%', SVC(2).perf.DA*100), ...
     'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('SVC with prior regression', 'FontSize', 16);
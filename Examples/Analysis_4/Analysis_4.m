% Multivariate Bayesian Inversion for Classification and Regression
% Analysis 4: brain age prediction (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 21/02/2022, 01:45: data analysis
% - 21/02/2022, 20:29: results visualization
% - 30/05/2022, 05:50: minor changes & finalization
% - 28/02/2025, 18:06: aligned with Python
% - 11/02/2026, 17:19: recorded analysis time


clear
close all

%%% Step 1: load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
load PAC_specify.mat
load PAC_specify_test_age.mat

% assemble data (MBR)
n1  = numel(sID1);              % number of data points
n2  = numel(sID2);
V1  = eye(n1);                  % observation covariances
V2  = eye(n2);
YA1 = [GM1, WM1];               % data matrices
YA2 = [GM2, WM2];
x1  = y1;                       % label vectors
x2  = y2;
XA1 = c1(:,2:end);              % covariate matrices
XA2 = c2(:,2:end);

% assemble data (SVR)
YB1 = [GM1, WM1, c1];           % feature matrices
YB2 = [GM2, WM2, c2];

% prepare histograms
x_min = 0;
x_max = 100;
dx    = 2.5;
xb    = [(x_min+dx/2):dx:(x_max-dx/2)];


%%% Step 2: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% estimate gamma distribution
y1_min = min(y1)-0.5;
ab_est = gamfit(y1-y1_min);

% define priors for MBR
priors = {'uniform', 'data-driven', 'fitted'};
prior(1).x = [0:1:100];
prior(1).p = (1/range(prior.x))*ones(size(prior.x));
prior(2).x = [0:1:100];
prior(2).p = hist(y1, prior(2).x)./trapz(prior(2).x, hist(y1, prior(2).x));
prior(3).x = [0:1:100];
prior(3).p = gampdf(prior(3).x-y1_min, ab_est(1), ab_est(2));

% Analysis 1: MBR with site/sex as covariates
tic;
fprintf('\n-> MBR: train, ');
MBA1 = mbitrain(YA1, x1, XA1, V1, 'MBR');
tA   = toc*ones(1,numel(prior));
PP2  = cell(1,numel(prior));
xMAP = zeros(n2,numel(prior));
rA   = zeros(1,numel(prior));
maeA = zeros(1,numel(prior));
fprintf('test: prior ');
for h = 1:numel(prior)
    tic;
    fprintf('%d, ', h);
    PP2{h} = mbitest(YA2, x2, XA2, V2, MBA1, prior(h));
    tA(h)  = tA(h) + toc;
    for i = 1:n2
        xMAP(i,h) = prior(h).x(PP2{h}(i,:)==max(PP2{h}(i,:)));
    end;
    rA(h)   = corr(xMAP(:,h), x2);
    maeA(h) = mean(abs(xMAP(:,h)-x2));
end;
fprintf('done.\n');

% Analysis 2: SVR with site/sex as features
tic;
fprintf('-> SVR: train, ');
SVM1 = svmtrain(x1, YB1, '-s 4 -t 0 -c 1 -q');
fprintf('test, ');
xp2  = svmpredict(x2, YB2, SVM1, '-q');
fprintf('done.\n\n');
tB   = toc;
rB   = corr(xp2, x2);
maeB = mean(abs(xp2-x2));

% calculate histograms
nb1 = hist(y1, xb);
nb2 = hist(y2, xb);
nbA = zeros(numel(prior), numel(xb));
nbB = hist(xp2,  xb);
for h = 1:numel(prior)
    nbA(h,:) = hist(xMAP(:,h), xb);
end;

% store analysis time
time = {'Analysis 4', 'Figure 11, 2nd/5th',  tA(1), tB;
        'Analysis 4', 'Figure 11, 3rd col.', tA(2), NaN;
        'Analysis 4', 'Figure 11, 4th col.', tA(3), NaN;}; 
save('Analysis_4.mat', 'time');


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Analysis 3 (1)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

% 1st row
subplot(3,5,1); hold on;
bar(xb, nb1, 'FaceColor', [3/4, 3/4, 3/4]);
plot(prior(3).x, prior(3).p*n1*dx, '-k', 'LineWidth', 2);
axis([x_min, x_max, 0, (11/10)*max(nb1)]);
set(gca,'Box','On');
xlabel('chronological age [yrs]', 'FontSize', 12);
ylabel('number of subjects', 'FontSize', 12);
title('Training Set', 'FontSize', 16);

for h = 1:numel(prior)
    subplot(3,5,1+h); hold on;
    plot([x_min, x_max], [x_min, x_max], '-k', 'LineWidth', 1);
    plot(x2, xMAP(:,h), '.b', 'MarkerSize', 10);
    axis([x_min, x_max, x_min, x_max]);
    axis square;
    set(gca,'Box','On');
    xlabel('actual age', 'FontSize', 12);
    ylabel('predicted age', 'FontSize', 12);
    if h == 2, title('MBR with site/gender as covariates', 'FontSize', 16); end;
    text(x_min+5, x_max-5, sprintf('r = %0.2f, MAE = %0.2f', rA(h), maeA(h)), ...
         'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Middle');
end;

subplot(3,5,5); hold on;
plot([x_min, x_max], [x_min, x_max], '-k', 'LineWidth', 1);
plot(x2, xp2, '.r', 'MarkerSize', 10);
axis([x_min, x_max, x_min, x_max]);
axis square;
set(gca,'Box','On');
xlabel('actual age', 'FontSize', 12);
ylabel('predicted age', 'FontSize', 12);
title('SVR with site/gender as features', 'FontSize', 16);
text(x_min+5, x_max-5, sprintf('r = %0.2f, MAE = %0.2f', rB, maeB), ...
     'HorizontalAlignment', 'Left', 'VerticalAlignment', 'Middle');

% 2nd row
subplot(3,5,6);
bar(xb, nb2, 'FaceColor', [3/4, 3/4, 3/4]);
axis([x_min, x_max, 0, (11/10)*max(nb2)]);
xlabel('chronological age [yrs]', 'FontSize', 12);
ylabel('number of subjects', 'FontSize', 12);
title('Validation Set', 'FontSize', 16);

for h = 1:numel(prior)
    subplot(3,5,6+h);
    bar(xb, nbA(h,:), 'b');
    axis([x_min, x_max, 0, (11/10)*max(nbA(h,:))]);
    xlabel('predicted age', 'FontSize', 12);
    ylabel('number of subjects', 'FontSize', 12);
    if h == 2, title('MBR: prediction distribution', 'FontSize', 16); end;
end;

subplot(3,5,10);
bar(xb, nbB, 'r');
axis([x_min, x_max, 0, (11/10)*max(nbB)]);
xlabel('predicted age', 'FontSize', 12);
ylabel('number of subjects', 'FontSize', 12);
title('SVR: prediction distribution', 'FontSize', 16);

% 3rd row
for h = 1:numel(prior)
    subplot(3,5,11+h);
    plot(prior(h).x, prior(h).p, '-b', 'LineWidth', 1);
    axis([x_min, x_max, 0, (11/10)*max(prior(h).p)]);
    if h == 1, ylim([0, 2*max(prior(h).p)]); end;
    xlabel('chronological age [yrs]', 'FontSize', 12);
    ylabel('prior density', 'FontSize', 12);
    title(sprintf('%s prior', priors{h}), 'FontSize', 16);
end;

% open figure
figure('Name', 'Analysis 3 (2)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

% all rows
for h = 1:numel(prior)
    for i = 1:5
        subplot(3,5,(h-1)*5+i); hold on;
        plot(x2(i), (1/10)*max(PP2{h}(i,:)), 'xk', 'MarkerSize', 7.5, 'LineWidth', 2);
        plot(xMAP(i,h), (1/10)*max(PP2{h}(i,:)), '.b', 'MarkerSize', 20);
        plot(prior(h).x, PP2{h}(i,:), '-b', 'LineWidth', 1);
        axis([min(prior(h).x), max(prior(h).x), 0, (11/10)*max(PP2{h}(i,:))]);
        set(gca,'Box','On');
        if h == 1 && i == 1, legend('true', 'mode', 'Location', 'NorthEast'); end;
        if h == 3, xlabel('chronological age [yrs]', 'FontSize', 12); end;
        if i == 1, ylabel(sprintf('%s prior', priors{h}), 'FontSize', 16, 'FontWeight', 'Bold'); end;
        if i == 2, ylabel('posterior density', 'FontSize', 12); end;
        if h == 1, title(sprintf('Subject %d', i), 'FontSize', 16); end;
    end;
end;
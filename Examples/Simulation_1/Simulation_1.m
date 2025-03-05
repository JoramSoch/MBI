% Multivariate Bayesian Inversion for Classification and Regression
% Simulation 1: two-class classification (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/02/2022, 12:23: first version
% - 07/07/2023, 21:17: minor changes
% - 17/02/2025, 14:13: reversed order for mu,
%                      corrected CA for s2,
%                      modified colormap
% - 18/02/2025, 16:29: aligned with Python


clear
close all

%%% Step 1: specify ground truth & model %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set ground truth
rng(3);
mu = [0:+0.25:1];               % class means
Si = [1, 0.5; 0.5, 1];          % covariance structure
s2 = [1:1:5].^2;                % noise variance
n  = 250;
k  = 10;
v  = 2;
C  = 2;

% generate classes
x  = [kron([1:C]',ones(n/C,1)), rand(n,1)];
x  = sortrows(x,2);
x  = x(:,1);
X  = zeros(n,C);
V  = eye(n);
for i = 1:n
    X(i,x(i)) = 1;
end;


%%% Step 2: generate & analyze the data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify cross-validation
CV = ML_CV(x, k, 'kfc');
CA = zeros(2,numel(mu));

% run simulations
for h = 1:numel(mu)
    
    % generate signals (variance fixed)
    B = [-mu(h), +mu(h);
         +mu(h), -mu(h)];
    E = matnrnd(zeros(n,v), s2(1)*V, Si, 1);
    Y = X*B + E;
    
    % cross-validated MBC
    MBC(1,h) = ML_MBI(Y, x, [], V, CV, 'MBC', []);
    CA(1,h)  = MBC(1,h).perf.CA;
    
    % generate signals (distance fixed)
    B = [-mu(end), +mu(end);
         +mu(end), -mu(end)];
    E = matnrnd(zeros(n,v), s2(h)*V, Si, 1);
    Y = X*B + E;
    
    % cross-validated MBC
    MBC(2,h) = ML_MBI(Y, x, [], V, CV, 'MBC', []);
    CA(2,h)  = MBC(2,h).perf.CA;
    
end;


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% create colormap
dc   = 0.01;
cmap = [[[0:dc:(1-dc)]', zeros(1/dc,1), ones(1/dc,1)];
        [1, 0, 1];
        [ones(1/dc,1), zeros(1/dc,1), [(1-dc):-dc:0]']];
lims = [4, 12];

% open figure
figure('Name', 'Simulation 1', 'Color', [1 1 1], 'Position', [50 50 2100 600]);

% plot results
for g = 1:2
    
    % plot features
    for h = 1:numel(mu)
        subplot(2,numel(mu)+1,(g-1)*(numel(mu)+1)+h); hold on;
        for i = 1:n
            plot(MBC(g,h).data.Y(i,1), MBC(g,h).data.Y(i,2), '.', 'MarkerSize', 10, ...
                 'Color', cmap(1+round(MBC(g,h).pred.PP(i,2)*((1+1)/dc)),:));
        end;
        axis([-lims(g), +lims(g), -lims(g), +lims(g)]);
        axis square;
        set(gca,'Box','On');
        if g == 2 && h == 1
            xlabel('feature 1', 'FontSize', 12);
            ylabel('feature 2', 'FontSize', 12);
        end;
        if g == 1, title(sprintf('distance: %2.2f', sqrt(8*mu(h)^2)), 'FontSize', 16); end;
        if g == 2, title(sprintf('std. dev.: %d', round(sqrt(s2(h)))), 'FontSize', 16); end;
    end;
    
    % plot accuracies
    subplot(2,numel(mu)+1,g*(numel(mu)+1));  hold on;
    if g == 1, x_gh = sqrt(8*mu.^2); end;
    if g == 2, x_gh = sqrt(s2); end;
    plot(x_gh, CA(g,:), ':ok', 'LineWidth', 2, 'MarkerSize', 2);
    xlim([min(x_gh)-0.1, max(x_gh)+0.1]);
    ylim([(0.5-0.05), (1+0.05)]);
    set(gca,'Box','On');
    ylabel('classification accuracy', 'FontSize', 12);
    if g == 1
        xlabel('distribution distance', 'FontSize', 12);
        title('std. dev. fixed', 'FontSize', 16);
    end;
    if g == 2
        xlabel('standard deviation', 'FontSize', 12);
        title('distance fixed', 'FontSize', 16);
    end;
end;
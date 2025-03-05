% Multivariate Bayesian Inversion for Classification and Regression
% Simulation 3: three-class classification (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/02/2022, 15:55: first version
% - 30/05/2022, 05:22: minor changes
% - 05/02/2022, 13:10: scaled linear RGB for better visualization
% - 17/02/2025, 12:24: linear RGB to sRGB for better visualization
% - 20/02/2025, 14:57: aligned with Python


clear
close all

%%% Step 1: specify ground truth %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set ground truth
rng(1);
mu = 1;                         % class means
Si = [1, 0.5; 0.5, 1];          % covariance structure
s2 = 4;                         % noise variance
n  = 300;
k  = 10;
v  = 2;
C  = 3;

% generate classes
x = [kron([1:C]',ones(n/C,1)), rand(n,1)];
x = sortrows(x,2);
x = x(:,1);
X = zeros(n,C);
V = eye(n);
for i = 1:n
    X(i,x(i)) = 1;
end;


%%% Step 2: generate & analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate data
B = [  -mu,   +mu;
     +2*mu, +2*mu;
       +mu,   -mu];
E = matnrnd(zeros(n,v), s2*V, Si, 1);
Y = X*B + E;

% specify cross-validation
CV = ML_CV(x, k, 'kfc');

% prepare prediction grid
lim = 6;
dxy = 0.05;
xy  = [(-lim+dxy/2):dxy:(+lim-dxy/2)];
x2  = [1:C, ones(1,numel(xy)-C)]';
n2  = numel(x2);

% Analysis 1: MBC
MBC = ML_MBI(Y, x, [], V, CV, 'MBC', []);
MBA = mbitrain(Y, x, [], V, 'MBC');

% Analysis 2: SVC
SVC = ML_SVC(x, Y, CV, 1, 1, 0);
SVM = svmtrain(x, Y, '-s 0 -t 0 -c 1 -q');

% Analysis 1: priors
DA  = [MBC.perf.DA, SVC.perf.DA];
DAp = zeros(1,C);
for k = 1:C
    prior.x    = [1:C];
    prior.p    = 1/6*ones(1,C);
    prior.p(k) = 2/3;
    MBC        = ML_MBI(Y, x, [], V, CV, 'MBC', prior);
    DAp(k)     = MBC.perf.DA;
end;

% Analysis 1 & 2: predictions
PP  = zeros(numel(xy),numel(xy),3);
Xp  = zeros(numel(xy),numel(xy),3);
PPp = zeros(numel(xy),numel(xy),3,C);
fprintf('-> Prediction grid:');
for i = 1:numel(xy)
    % specify test data
    if mod(i,10) == 1
        fprintf('\n   - x = ');
    end;
    fprintf('%0.3f, ', xy(i));
    Y2 = [xy(i)*ones(size(xy')), xy'];
    % MBC: posterior probabilities
    pp = mbitest(Y2, x2, [], eye(n2), MBA, []);
    for j = 1:C
        PP(:,i,j) = pp(:,j);
    end;
    % SVC: predicted classes
    xp = svmpredict(x2, Y2, SVM, '-q');
    for j = 1:C
        Xp(xp==j,i,j) = 1;
    end;
    % MBC: modified priors
    prior.x = [1:C];
    for j1 = 1:C
        prior.p     = 1/6*ones(1,C);
        prior.p(j1) = 2/3;
        pp = mbitest(Y2, x2, [], eye(n2), MBA, prior);
        for j2 = 1:C
            PPp(:,i,j2,j1) = pp(:,j2);
        end;
    end;
end;
fprintf('done.\n\n');
clear pp xp prior

% edit posterior probabilities
thr =  0.0031308;
PP  = 12.92*PP.*(PP<=thr)   + 1.055*(PP.^(1/2.4)).*(PP>thr);
PPp = 12.92*PPp.*(PPp<=thr) + 1.055*(PPp.^(1/2.4)).*(PPp>thr);
% Source: https://en.wikipedia.org/w/index.php?title=SRGB&oldid=1226800876#From_CIE_XYZ_to_sRGB
PP  = uint8(PP*255);
Xp  = uint8(Xp*255);
PPp = uint8(PPp*255);


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Simulation 3', 'Color', [1 1 1], 'Position', [50 50 1600 900]);

% plot features
subplot(2,3,1); hold on;
plot(Y(x==1,1), Y(x==1,2), '.r', 'MarkerSize', 10);
plot(Y(x==2,1), Y(x==2,2), '.g', 'MarkerSize', 10);
plot(Y(x==3,1), Y(x==3,2), '.b', 'MarkerSize', 10);
axis([-lim, +lim, -lim, +lim]);
axis square;
set(gca,'Box','On');
legend('class 1', 'class 2', 'class 3', 'Location', 'SouthEast');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('Training Data', 'FontSize', 16);

% plot MBC posterior probabilities
subplot(2,3,2);
imagesc(xy,xy,PP);
axis([-lim, +lim, -lim, +lim]);
axis xy square;
set(gca,'Box','On');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('MBC: posterior probabilities', 'FontSize', 16);
text(+(9/10)*lim, -(9/10)*lim, sprintf('CA = %2.2f %%', DA(1)*100), ...
     'Color', 'w', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');

% plot SVC predicted classes
subplot(2,3,3);
imagesc(xy,xy,Xp);
axis([-lim, +lim, -lim, +lim]);
axis xy square;
set(gca,'Box','On');
xlabel('feature 1', 'FontSize', 12);
ylabel('feature 2', 'FontSize', 12);
title('SVC: predicted classes', 'FontSize', 16);
text(+(9/10)*lim, -(9/10)*lim, sprintf('CA = %2.2f %%', DA(2)*100), ...
     'Color', 'w', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');

% plot MBC with modified priors
for k = 1:C
    subplot(2,3,3+k);
    imagesc(xy,xy,PPp(:,:,:,k));
    axis([-lim, +lim, -lim, +lim]);
    axis xy square;
    set(gca,'Box','On');
    xlabel('feature 1', 'FontSize', 12);
    ylabel('feature 2', 'FontSize', 12);
    title(sprintf('MBC: class %d more likely a priori', k), 'FontSize', 16);
    text(+(9/10)*lim, -(9/10)*lim, sprintf('CA = %2.2f %%', DAp(k)*100), ...
         'Color', 'w', 'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');
end;
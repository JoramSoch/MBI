% Multivariate Bayesian Inversion for Classification and Regression
% Analysis 3: birth weight data (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 21/02/2022, 19:10: first version
% - 30/05/2022, 05:33: minor changes
% - 27/02/2025, 11:27: aligned with Python
% - 11/02/2026, 16:59: recorded analysis time


clear
close all

%%% Step 1: load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load CSV file
filename = 'Birth_Weights.csv';
fid = fopen(filename);
raw = textscan(fid,'%f%f%d%s%s%s%d','HeaderLines',1,'Delimiter',',');
hdr = {'birth weight','mother''s weight','mother''s age','smoker','ethnicity','hypertension','visits to the doctor'};
fclose(fid);

% extract data
Y = [raw{1}, raw{2}];           % data matrix
X = zeros(size(Y,1),5);         % design matrix
X(:,1) = 1*strcmp(raw{4},'"no"') + 2*strcmp(raw{4},'"yes"');
X(:,2) = 1*strcmp(raw{5},'"white"') + 2*strcmp(raw{5},'"black"') + 3*strcmp(raw{5},'"other"');
X(:,3) = 1*strcmp(raw{6},'"no"') + 2*strcmp(raw{6},'"yes"');
X(:,4) = raw{3};
X(:,5) = raw{7};
n = size(Y,1);                  % number of data points
v = size(Y,2);                  % number of features


%%% Step 2: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify cross-validation
k  = 10;                        % number of CV folds
V  = eye(n);                    % observation covariance
tA = 0;
tB = 0;

% Analysis 1: classify smoker, (not) accounting for others
x1  = X(:,1);
% X1  = [1*(X(:,2)==1)-1*(X(:,2)>1), 1*(X(:,3)==1)-1*(X(:,3)==2), X(:,4:5)];
X1  = [1*(X(:,2)==1)-1*(X(:,2)==2), 1*(X(:,2)==2)-1*(X(:,2)==3), 1*(X(:,3)==1)-1*(X(:,3)==2), X(:,4:5)];
% X1  = [1*(X(:,2)==1)-1*(X(:,2)==2), 1*(X(:,2)==2)-1*(X(:,2)==3), 1*(X(:,3)==1)-1*(X(:,3)==2)];
% X1  = [X(:,4:5)];
CV1 = ML_CV(x1, k, 'kfc');
X1r = [X1, ones(n,1)];
Y1r = (eye(n) - X1r*(X1r'*X1r)^(-1)*X1r')*Y;
tic; MBC(1) = ML_MBI(Y, x1, X1, V, CV1, 'MBC', []); tA = tA + toc;
tic; SVC(1) = ML_SVC(x1, Y1r, CV1, 1, 1, 0);        tB = tB + toc;

% Analysis 2: classify ethnicity, (not) accounting for others
x2  = X(:,2);
X2  = [1*(X(:,1)==1)-1*(X(:,1)==2), 1*(X(:,3)==1)-1*(X(:,3)==2), X(:,4:5)];
% X2  = [1*(X(:,1)==1)-1*(X(:,1)==2), 1*(X(:,3)==1)-1*(X(:,3)==2)];
% X2  = [X(:,4:5)];
CV2 = ML_CV(x2, k, 'kfc');
X2r = [X2, ones(n,1)];
Y2r = (eye(n) - X2r*(X2r'*X2r)^(-1)*X2r')*Y;
tic; MBC(2) = ML_MBI(Y, x2, X2, V, CV2, 'MBC', []); tA = tA + toc;
tic; SVC(2) = ML_SVC(x2, Y2r, CV2, 1, 1, 0);        tB = tB + toc;

% Analysis 3: classify hypertension, (not) accounting for others
x3  = X(:,3);
% X3  = [1*(X(:,1)==1)-1*(X(:,1)==2), 1*(X(:,2)==1)-1*(X(:,2)>1), X(:,4:5)];
X3  = [1*(X(:,1)==1)-1*(X(:,1)==2), 1*(X(:,2)==1)-1*(X(:,2)==2), 1*(X(:,2)==2)-1*(X(:,2)==3), X(:,4:5)];
% X3  = [1*(X(:,1)==1)-1*(X(:,1)==2), 1*(X(:,2)==1)-1*(X(:,2)==2), 1*(X(:,2)==2)-1*(X(:,2)==3)];
% X3  = [X(:,4:5)];
CV3 = ML_CV(x3, 6, 'kfc');
X3r = [X3, ones(n,1)];
Y3r = (eye(n) - X3r*(X3r'*X3r)^(-1)*X3r')*Y;
tic; MBC(3) = ML_MBI(Y, x3, X3, V, CV3, 'MBC', []); tA = tA + toc;
tic; SVC(3) = ML_SVC(x3, Y3r, CV3, 1, 1, 0);        tB = tB + toc;
% collect balanced accuracies
BA = zeros(2,3);
for h = 1:3
    BA(1,h) = MBC(h).perf.BA;
    BA(2,h) = SVC(h).perf.BA;
end;

% store analysis time
time = {'Analysis 3', 'Figure 10B', tA, tB}; 
save('Analysis_3.mat', 'time');


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Analysis 2 (B)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
cmap2= [repmat([0:0.01:1]',[1 3]); [[0.99:-0.01:0]',ones(100,1),[0.99:-0.01:0]']];
cmap3= [repmat([0:0.02:1]',[1 3]); [[0.99:-0.01:0]',ones(100,1),[0.99:-0.01:0]']];
comp = {'smok.','ethn.','tension'};
labs ={{'non-smoker','smoker'}, ...
       {'white','black','other'}, ...
       {'normal tension','hypertension'}};

% confusion matrices
nC = zeros(1,numel(labs));
for h = 2 % 1:numel(labs)
    nC(h) = max(X(:,h));
    for g = 1:2
        subplot(3,4,(h-1)*4+(2+g));
        if g == 1, CM = MBC(h).perf.CM; end;
        if g == 2, CM = SVC(h).perf.CM; end;
        imagesc(CM);
        caxis([0, 1]);
        colormap(cmap3);
        colorbar;
        axis ij;
        set(gca,'XTick',[1:nC(h)],'XTickLabel',labs{h});
        set(gca,'YTick',[1:nC(h)]);
        xlabel('true class', 'FontSize', 12);
        ylabel('predicted class', 'FontSize', 12);
        if h == 1
            if g == 1, title('MBC with covariates', 'FontSize', 16); end;
            if g == 2, title('SVC with regression', 'FontSize', 16); end;
        end;
        for c1 = 1:nC(h)
            for c2 = 1:nC(h)
                text(c1, c2, sprintf('%0.2f', CM(c2,c1)), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
            end;
        end;
    end;
end;

% open figure
figure('Name', 'Analysis 2 (A)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
cols =  'brgcm';

% data set (classes)
for k = 1:numel(labs)
    subplot(3,6,(k-1)*6+1); hold on;
    str = 'N = ';
    for j = 1:max(X(:,k))
        plot(Y(X(:,k)==j,2), Y(X(:,k)==j,1), strcat('.',cols(j)), 'MarkerSize', 10);
        str = sprintf('%s%d, ', str, sum(X(:,k)==j));
    end;
    xlim([min(Y(:,2))-(1/20)*range(Y(:,2)), max(Y(:,2))+(1/20)*range(Y(:,2))]);
    ylim([min(Y(:,1))-(1/20)*range(Y(:,1)), max(Y(:,1))+(1/20)*range(Y(:,1))]);
    set(gca,'Box','On');
    legend(labs{k}, 'Location', 'NorthEast');
    xlabel(hdr{2}, 'FontSize', 12);
    ylabel(hdr{1}, 'FontSize', 12);
    if k == 1, title('Data Set', 'FontSize', 16); end;
    text(max(Y(:,2)), min(Y(:,1)), str(1:end-2), 'FontSize', 10, ...
         'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Bottom');
end;

% data set (targets)
for k1 = 1:2
    for k2 = 1:2
        subplot(3,6,6+(k2-1)*6+1+k1); hold on;
        xk = X(:,3+k1);
        yk = Y(:,k2);
        plot(xk, yk, strcat('.',cols(3+k1)), 'MarkerSize', 10);
        xlim([min(xk)-(1/20)*range(xk), max(xk)+(1/20)*range(xk)]);
        ylim([min(yk)-(1/20)*range(yk), max(yk)+(1/20)*range(yk)]);
        set(gca,'Box','On');
        if k1 == 1, xlabel(hdr{3}, 'FontSize', 12); end;
        if k1 == 2, xlabel(hdr{7}, 'FontSize', 12); end;
        ylabel(hdr{k2}, 'FontSize', 12);
        text(max(xk), max(yk), sprintf('r = %0.2f', corr(xk,yk)), 'FontSize', 10, ...
             'HorizontalAlignment', 'Right', 'VerticalAlignment', 'Top');
    end;
end;

% confusion matrices
nC = zeros(1,numel(labs));
for h = [1,3] % 1:numel(labs)
    nC(h) = max(X(:,h));
    for g = 1:2
        subplot(3,4,(h-1)*4+(2+g));
        if g == 1, CM = MBC(h).perf.CM; end;
        if g == 2, CM = SVC(h).perf.CM; end;
        imagesc(CM);
        caxis([0, 1]);
        colormap(cmap2);
        colorbar;
        axis ij;
        set(gca,'XTick',[1:nC(h)],'XTickLabel',labs{h});
        set(gca,'YTick',[1:nC(h)]);
        xlabel('true class', 'FontSize', 12);
        ylabel('predicted class', 'FontSize', 12);
        if h == 1
            if g == 1, title('MBC with covariates', 'FontSize', 16); end;
            if g == 2, title('SVC with regression', 'FontSize', 16);    end;
        end;
        for c1 = 1:nC(h)
            for c2 = 1:nC(h)
                text(c1, c2, sprintf('%0.2f', CM(c2,c1)), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
            end;
        end;
    end;
end;

% classification accuracies
subplot(3,6,3); hold on;
bp = bar([1:size(BA,2)], BA', 'grouped');
set(bp(1), 'FaceColor', 'b');
set(bp(2), 'FaceColor', 'r');
plot([0, 1.5, 1.5, 2.5, 2.5, 4], [1/2, 1/2, 1/3, 1/3, 1/2, 1/2], ':k', 'LineWidth', 2);
axis([(1-1), (size(BA,2)+1), 0, 1]);
set(gca,'Box','On');
set(gca,'XTick',[1:size(BA,2)],'XTickLabel',comp);
legend('MBC', 'SVC', 'chance', 'Location', 'NorthWest');
xlabel('classified variable', 'FontSize', 12);
ylabel('balanced accuracy', 'FontSize', 12);
title('Classification', 'FontSize', 16);
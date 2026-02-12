% Multivariate Bayesian Inversion for Classification and Regression
% Analysis 1: Egyptian skull data (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 21/02/2022, 00:41: first version
% - 21/02/2022, 17:18: minor changes
% - 25/02/2025, 14:28: aligned with Python
% - 11/02/2026, 16:43: recorded analysis time


clear
close all

%%% Step 1: load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load TSV file
filename = 'Egyptian_Skulls.tsv';
[data, hdr, raw] = tsvread(filename);

% extract data
data = data(~isnan(data(:,5)),:);
xC   = unique(data(:,5))';      % class labels
x    = data(:,5);               % label vector
Y    = data(:,1:4);             % data matrix
n    = size(Y,1);               % number of data points
v    = size(Y,2);               % number of features

% assign classes
C = numel(xC);
for j = 1:C                     % replace class labels
    x(x==xC(j)) = j;            % by 1, 2, 3, ...
end;


%%% Step 2: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify cross-validation
k  = 10;                        % number of CV folds
V  = eye(numel(x));             % observation covariance
CV = ML_CV(x, k, 'kfc');        % n x k CV matrix

% specify analyses
iC{1} = [1:5];
it{1} = ismember(x,iC{1});
xt{1} = x(it{1});
iC{2} = [1,3,5];
it{2} = ismember(x,iC{2});
xt{2} = x(it{2}); xt{2}(xt{2}==3) = 2; xt{2}(xt{2}==5) = 3; 
iC{3} = [1,5];
it{3} = ismember(x,iC{3});
xt{3} = x(it{3}); xt{3}(xt{3}==5) = 2;

% Analyses 1: MBC
tic;
for h = 1:numel(it)
    MBC(h) = ML_MBI(Y(it{h},:), xt{h}, [], V(it{h},it{h}), CV(it{h},:), 'MBC', []);
end;
tA = toc;

% Analyses 2: SVM
tic;
for h = 1:numel(it)
    SVC(h) = ML_SVC(xt{h}, Y(it{h},:), CV(it{h},:), 1, 1, 0);
end;
tB = toc;

% store analysis time
time = {'Analysis 1', 'Figure 7B', tA, tB}; 
save('Analysis_1.mat', 'time');


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Analysis 1', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
cmap = [repmat([0:0.01:1]',[1 3]); [[0.99:-0.01:0]',ones(100,1),[0.99:-0.01:0]']];
labs = cellstr(num2str(xC'))';
cols = 'rgbcm';

% data set
for k1 = 1:(v-1)
    for k2 = (k1+1):v
        subplot(v-1,2*(v-1),(k2-2)*2*(v-1)+k1); hold on;
        for j = 1:C
            plot(Y(x==j,k1), Y(x==j,k2), strcat('.',cols(j)), 'MarkerSize', 10);
        end;
        xlim([min(Y(:,k1))-(1/20)*range(Y(:,k1)), max(Y(:,k1))+(1/20)*range(Y(:,k1))]);
        ylim([min(Y(:,k2))-(1/20)*range(Y(:,k2)), max(Y(:,k2))+(1/20)*range(Y(:,k2))]);
        set(gca,'Box','On');
        if (k1 == 1 && k2 == 2) || (k1 == 2 && k2 == 3)
            legend(labs, 'Location', 'SouthEast');
        end;
        xlabel(hdr{k1}, 'FontSize', 12);
        ylabel(hdr{k2}, 'FontSize', 12);
        if k1 == 1 && k2 == 2
            title('Data Set', 'FontSize', 16);
        end;
    end;
end;

% confusion matrices
nC = zeros(1,numel(iC));
CA = zeros(2,numel(xt));
for h = 1:numel(xt)
    nC(h) = numel(iC{h});
    for g = 1:2
        subplot(3,4,(h-1)*4+(2+g));
        if g == 1, CM = MBC(h).perf.CM; end;
        if g == 2, CM = SVC(h).perf.CM; end;
        imagesc(CM);
        caxis([0 2*(1/nC(h))]);
        colormap(cmap);
        colorbar;
        axis ij;
        set(gca,'XTick',[1:nC(h)],'XTickLabel',labs(iC{h}));
        set(gca,'YTick',[1:nC(h)],'YTickLabel',labs(iC{h}));
        xlabel('true class', 'FontSize', 12);
        ylabel('predicted class', 'FontSize', 12);
        if g == 1, title(sprintf('MBC: %d classes', nC(h)), 'FontSize', 16); end;
        if g == 2, title(sprintf('SVC: %d classes', nC(h)), 'FontSize', 16); end;
        for c1 = 1:max(xt{h})
            for c2 = 1:max(xt{h})
                text(c1, c2, sprintf('%0.2f', CM(c2,c1)), 'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
            end;
        end;
        if g == 1, CA(g,h) = MBC(h).perf.CA; end;
        if g == 2, CA(g,h) = SVC(h).perf.DA; end;
    end;
end;

% classification accuracies
subplot(v-1,2*(v-1),3); hold on;
bp = bar([1:numel(nC)], CA', 'grouped');
set(bp(1), 'FaceColor', 'b');
set(bp(2), 'FaceColor', 'r');
plot([0, 1.5, 1.5, 2.5, 2.5, 4], [1/5, 1/5, 1/3, 1/3, 1/2, 1/2], ':k', 'LineWidth', 2);
axis([(1-1), (numel(nC)+1), 0, 1]);
set(gca,'Box','On');
set(gca,'XTick',[1:numel(nC)],'XTickLabel',cellstr(num2str(nC'))');
legend('MBC', 'SVC', 'chance', 'Location', 'NorthWest');
xlabel('number of classes', 'FontSize', 12);
ylabel('classification accuracy', 'FontSize', 12);
title('Classification', 'FontSize', 16);
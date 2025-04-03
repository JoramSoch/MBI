% Multivariate Bayesian Inversion for Classification and Regression
% Analysis 2: MNIST digit recognition (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/12/2024, 17:46: data analysis
% - 03/01/2025, 16:57: results visualization
% - 08/01/2025, 21:44: minor changes & finalization
% - 26/02/2025, 16:04: aligned with Python


clear
close all

% define steps
steps = [1, 2, 3];


%%% Step 1: load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ismember(1,steps)

% load training data
n1  = 60000;
fnY = 'train-images.idx3-ubyte';
fnx = 'train-labels.idx1-ubyte';
[imgs1, labs1] = readMNIST(fnY, fnx, n1, 0);

% load test data
n2  = 10000;
fnY = 't10k-images.idx3-ubyte';
fnx = 't10k-labels.idx1-ubyte';
[imgs2, labs2] = readMNIST(fnY, fnx, n2, 0);

% extract data
v  = size(imgs1,1)*size(imgs1,2);%number of features
Y1 = reshape(imgs1, [v,n1])';   % training data matrix
Y2 = reshape(imgs2, [v,n2])';   % test data matrix
x1 = labs1;                     % training labels
x2 = labs2;                     % test labels
x1(x1==0) = 10;                 % replace 0 by 10
x2(x2==0) = 10;
clear fnY fnx imgs1 labs1 imgs2 labs2

% save extracted data
save('MNIST_data.mat', 'Y1', 'Y2', 'x1', 'x2');

end;


%%% Step 2: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ismember(2,steps)

% load extracted data
load('MNIST_data.mat');

% specify analyses
N1 =[600, 1000:1000:10000, 12000:2000:20000, 25000:5000:numel(x1)];
N2 =[1000, numel(x2)];

% preallocate results
CA_MBC = zeros(numel(N2),numel(N1));
CA_SVC = zeros(numel(N2),numel(N1));
fprintf('\n-> Train and test on MNIST data set:\n');

% loop over training data points
for i = 1:numel(N1)
    
    % get number of data points
    n1 = N1(i);
    fprintf('   - n1 = %d:\n', n1);
    
    % MBC: training
    fprintf('     - training: MBC ... ');
    MBA1 = mbitrain(Y1(1:n1,:), x1(1:n1), [], speye(n1), 'MBC');
    fprintf('successful!\n');
    
    % SVC: training
    fprintf('     - training: SVC ... ');
    SVM1 = svmtrain(x1(1:n1), Y1(1:n1,:), '-s 0 -t 0 -c 1 -q');
    fprintf('successful!\n');
    
    % loop over test data points
    for j = 1:numel(N2)
        
        % get number of data points
        n2 = N2(j);
        fprintf('     - n2 = %d:\n', n2);
        
        % MBC: testing
        fprintf('       - testing: MBC ... ');
        PP2         = mbitest(Y2(1:n2,:), x2(1:n2), [], speye(n2), MBA1, []);
       [PP_max, xp] = max(PP2, [], 2);
        CA_MBC(j,i) = mean(xp==x2(1:n2));
        fprintf('successful!\n');
        
        % SVC: testing
        fprintf('       - testing: SVC ... ');
        xp2         = svmpredict(x2(1:n2), Y2(1:n2,:), SVM1, '-q');
        CA_SVC(j,i) = mean(xp2==x2(1:n2));
        fprintf('successful!\n');
        
    end;
    
end;

% save analysis results
fprintf('\n');
save('MNIST_analysis.mat', 'N1', 'N2', 'MBA1', 'PP2', 'SVM1', 'xp2', 'CA_MBC', 'CA_SVC');

end;


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if ismember(3,steps)

% load analysis results
load('MNIST_data.mat');
load('MNIST_analysis.mat');
nC = max(x1);

% create confusion matrices
[PP_max, xp] = max(PP2, [], 2);
CM_MBC = zeros(nC,nC);
CM_SVC = zeros(nC,nC);
for j = 1:nC
    CM_MBC(:,j) = mean(repmat(xp(x2==j),  [1,nC])==repmat([1:nC], [sum(x2==j),1]))';
    CM_SVC(:,j) = mean(repmat(xp2(x2==j), [1,nC])==repmat([1:nC], [sum(x2==j),1]))';
end;
CM_MBC = CM_MBC([nC, 1:(nC-1)], [nC, 1:(nC-1)]);
CM_SVC = CM_SVC([nC, 1:(nC-1)], [nC, 1:(nC-1)]);

% calculate proportions correct
dx    = 0.04;
alpha = 0.05;
xe_PP =[0:dx:1];
xc_PP =[(dx/2):dx:(1-(dx/2))];
f_mean= zeros(1,numel(xc_PP));
f_CI  = zeros(2,numel(xc_PP));
for j = 1:numel(xc_PP)
    i_PP = PP_max>xe_PP(j) & PP_max<=xe_PP(j+1);
    n_PP = sum(i_PP);
    if n_PP > 0
        n_CA         = sum(xp(i_PP)==x2(i_PP));
       [p_hat, p_CI] = binofit(n_CA, n_PP, alpha);
        f_mean(j)    = p_hat;
        f_CI(:,j)    = p_CI';
    else
        f_mean(j)    = NaN;
        f_CI(:,j)    =[NaN; NaN];
    end;
end;
clear dx xe_PP i_PP n_PP n_CA p_hat p_CI

% extract precision matrices
O1 = MBA1.post.O1;
L1 = MBA1.post.L1;
L1 = L1([nC, 1:(nC-1)], [nC, 1:(nC-1)]);
    
% open figure
figure('Name', 'Analysis MNIST (1A)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
cmap = [repmat([0:(1/9):1]',[1 3]); [[(89/90):-(1/90):0]',ones(90,1),[(89/90):-(1/90):0]']];
labs = cellstr(num2str(([1:nC]-1)'))';

% classification accuracies
subplot(2,3,1); hold on;
plot(N1, CA_MBC(end,:), '-b', 'LineWidth', 2);
plot(N1, CA_SVC(end,:), '-r', 'LineWidth', 2);
plot([0, max(N1)], [1/nC, 1/nC], ':k', 'LineWidth', 2);
axis([0, max(N1), 0, 1]);
set(gca,'Box','On');
legend({'MBC', 'SVC', 'chance'}, 'Location', 'East');
xlabel('number of training samples', 'FontSize', 12);
ylabel('classification accuracy', 'FontSize', 12);
title('Classification', 'FontSize', 14);

% confusion matrices
for g = 1:2
    subplot(2,3,1+g);
    if g == 1, CM = CM_MBC; end;
    if g == 2, CM = CM_SVC; end;
    imagesc(CM);
    caxis([0 1]);
    colormap(cmap);
    colorbar;
    axis ij; % square;
    set(gca,'XTick',[1:nC],'XTickLabel',labs);
    set(gca,'YTick',[1:nC],'YTickLabel',labs);
    xlabel('true class', 'FontSize', 12);
    ylabel('predicted class', 'FontSize', 12);
    if g == 1, title(sprintf('MBC: %d classes', nC), 'FontSize', 14); end;
    if g == 2, title(sprintf('SVC: %d classes', nC), 'FontSize', 14); end;
    for c1 = 1:nC
        for c2 = 1:nC
            text(c1, c2, sprintf('%0.2f', CM(c2,c1)), 'FontSize', 9, ...
                'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Middle');
        end;
    end;
end;

% open figure
figure('Name', 'Analysis MNIST (1B)', 'Color', [1 1 1], 'Position', [50 50 1600 900]);
cmap = [[repmat([0:0.01:0.99]',[1 2]), ones(100,1)]; [1 1 1]; [ones(100,1), repmat([0.99:-0.01:0]',[1 2])]];

% maximum posterior probability
subplot(2,3,4); hold on;
plot(xc_PP, f_mean, 'ob', 'LineWidth', 2, 'MarkerSize', 5, 'MarkerFaceColor', 'b');
errorbar(xc_PP, f_mean, (f_mean-f_CI(1,:)), (f_CI(2,:)-f_mean), ...
         '.b', 'LineWidth', 2, 'CapSize', 10);
plot([0,1], [0,1], ':k', 'LineWidth', 2);
axis([0, 1, 0, 1]);
set(gca,'Box','On');
set(gca,'XTick',[0:0.1:1]);
set(gca,'YTick',[0:0.1:1]);
legend({'average', '95% CI', 'identity'}, 'Location', 'NorthWest');
xlabel('posterior probability of most likely class', 'FontSize', 12);
ylabel('frequency of most likely being true class', 'FontSize', 12);
title('Frequency vs. Probability', 'FontSize', 14);

% posterior inverse scale matrix
O1_max = max(max(abs(O1)));
subplot(2,3,5);
imagesc(O1);
caxis([-O1_max, +O1_max]);
% colormap(cmap);
axis ij square;
set(gca,'XTick',[50:50:size(Y1,2)]);
set(gca,'YTick',[50:50:size(Y1,2)]);
xlabel('image pixel', 'FontSize', 12);
ylabel('image pixel', 'FontSize', 12);
title('MBC: posterior inverse scale matrix', 'FontSize', 14);

% posterior precision matrix
L1_max = max(max(abs(L1)));
subplot(2,3,6);
imagesc(L1);
caxis([-L1_max, +L1_max]);
% colormap(cmap);
axis ij square;
set(gca,'XTick',[1:nC],'XTickLabel',labs);
set(gca,'YTick',[1:nC],'YTickLabel',labs);
xlabel('digit category', 'FontSize', 12);
ylabel('digit category', 'FontSize', 12);
title('MBC: posterior precision matrix', 'FontSize', 14);

% open figure
figure('Name', 'Analysis MNIST (2)', 'Color', [1 1 1], 'Position', [50 50 2000 600]);
labs = cellstr(num2str([1:9,0]'))';
w    = round(sqrt(size(Y1,2)));

% test examples
for j = 1:nC
    for g = 1:3
        % select image
        if g < 3
            Y_j  = Y2(x2==j & xp==j,:);         % correctly predicted
            xp_j = xp(x2==j & xp==j,:);         % predicted class
            PP_j = PP_max(x2==j & xp==j);       % posterior probability
        else
            Y_j  = Y2(x2==j & xp~=j,:);         % incorrectly predicted
            xp_j = xp(x2==j & xp~=j,:);         % predicted class
            PP2j = PP2(x2==j & xp~=j,:);        % posterior probabilities
            PP_j = zeros(size(Y_j,1),1);        % posterior probability
            for i = 1:size(Y_j,1)
                PP_j(i) = PP2j(i,xp_j(i));
            end;
            clear PP2j
        end;
        if g == 1
           [PPj_max, i] = max(PP_j);            % high-confidence hit
        elseif g == 2
           [PPj_min, i] = min(PP_j);            % low-confidence hit
        elseif g == 3
            i = 1;                              % randomly correct
         % [PPj_max, i] = max(PP_j);            % high-confidnce miss
        end;
        xp_i = xp_j(i(1));
        pp_i = PP_j(i(1));
        Y_i  = reshape(Y_j(i(1),:), [w,w]);
        % plot image
        if j < nC, subplot(3,nC,(g-1)*nC+j+1);
        else,      subplot(3,nC,(g-1)*nC+1);   end;
        imagesc(Y_i);
        caxis([0 1]);
        set(gca,'Box','On');
        set(gca,'XTick',[],'YTick',[]);
        if j == nC
            if g == 1, ylabel('maximum PP',           'FontSize', 12, 'FontWeight', 'Bold'); end;
            if g == 2, ylabel('minimum PP',           'FontSize', 12, 'FontWeight', 'Bold'); end;
            if g == 3, ylabel('incorrect prediction', 'FontSize', 12, 'FontWeight', 'Bold'); end;
        end;
        title(sprintf('PP(''%s'') = %0.2f', labs{xp_i}, pp_i), 'FontSize', 12);
    end;
end;
clear Y_j PP_j PPj_max PPj_min xp_i pp_i Y_i


end;
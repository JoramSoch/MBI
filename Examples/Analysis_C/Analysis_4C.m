% Multivariate Bayesian Inversion for Classification and Regression
% Analysis 4: comparison of methods (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 21/02/2022, 01:45: first data analysis
% - 28/02/2025, 18:06: final figure display
% - 26/02/2026, 17:13: rewrote for multiple analyses
% - 26/02/2026, 17:28: added MBR, GNB, MLR, SVR, RFR, NNR


clear
close all

%%% Step 1: load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
load ../Analysis_4/PAC_specify.mat
load ../Analysis_4/PAC_specify_test_age.mat

% assemble data (MBR)
n1  = numel(sID1);              % number of data points
n2  = numel(sID2);
V1  = eye(n1);                  % observation covariances
V2  = eye(n2);
YA1 = [GM1, WM1];               % data matrices
YA2 = [GM2, WM2];
x1  = y1;                       % label vectors
x2  = y2;
X1  = [x1, ones(size(x1))];     % design matrices
X2  = [x2, ones(size(x2))];
XA1 = c1(:,2:end);              % covariate matrices
XA2 = c2(:,2:end);

% assemble data (SVR)
YB1 = [GM1, WM1, c1];           % feature matrices
YB2 = [GM2, WM2, c2];


%%% Step 2: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify analysis parameters
meth    = {'MBR', 'GNB', 'MLR', 'SVR', 'RFR', 'NNR'};
prior.x = [0:1:100];            % MBR prior distribution
prior.p = (1/range(prior.x))*ones(size(prior.x));
Dgnb    = 'mvn';                % GNB distribution name
Mmlr    = 'WLS';                % MLR estimation method
Csvm    = 1;                    % SVM cost parameter
Mrf     = 'Bag';                % RF aggregation method
Lnn     = [size(YB1,2)];        % NN hidden layer sizes
Ann     = 'none';               % NN activation function

% preallocate results
M   = numel(meth);
xp  = zeros(n2,M);
r   = zeros( 1,M);
MAE = zeros( 1,M);

% evaluate all methods
for h = 1:M
    
    % Method: multivariate Bayesian regression (MBI)
    % https://github.com/JoramSoch/MBI/blob/main/MATLAB/ML_MBI.m
    if strcmp(meth{h}, 'MBR')
        mba1 = mbitrain(YA1, x1, XA1, V1, 'MBR');
        pp2  = mbitest(YA2, x2, XA2, V2, mba1, prior);
        [m,j]= max(pp2, [], 2);
        xp2  = prior.x(j);
    end;
    
    % Method: Gaussian naive Bayes (custom)
    % https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model
    if strcmp(meth{h}, 'GNB')
        B_est1  = (X1'*X1)^-1 * X1'*YA1;
        s2_est1 = (1/n1) * sum((YA1-X1*B_est1).^2);
        Si_est1 = diag(s2_est1);
        log_PP  = zeros(n2,numel(prior.x));
        pp2     = zeros(n2,numel(prior.x));
        for i = 1:n2
            y2i = YA2(i,:);
            for j = 1:numel(prior.x)
                x2ij        = [prior.x(j), 1];
                log_PP(i,j) = log(mvnpdf(y2i, x2ij*B_est1, Si_est1)) + ...
                              log(prior.p(j));
            end;
            pp2(i,:) = exp(log_PP(i,:) - mean(log_PP(i,:)));
            pp2(i,:) = pp2(i,:)./trapz(prior.x, pp2(i,:));
        end;
        [m,j]= max(pp2, [], 2);
        xp2  = prior.x(j);
    end;
    
    % Method: multiple linear regression (MACS)
    % https://github.com/JoramSoch/MACS/blob/master/ME_GLM.m
    if strcmp(meth{h}, 'MLR')
        P1  = inv(V1);
        if strcmp(Mmlr, 'OLS')
            b_est1 = (YB1'*YB1)^-1 * YB1'*x1;
        elseif strcmp(Mmlr, 'WLS')
            b_est1 = (YB1'*P1*YB1)^-1 * YB1'*P1*x1;
        else
            b_est1 = zeros(size(YB1,2),1);
        end;
        xp2 = YB2 * b_est1;
    end;
    
    % Method: support vector regression (LibSVM)
    % https://github.com/JoramSoch/ML4ML/blob/main/ML_SVR.m
    if strcmp(meth{h}, 'SVR')
        opt  = sprintf('-s 4 -t 0 -c %s -q', num2str(Csvm));
        svm1 = svmtrain(x1, YB1, opt);
        xp2  = svmpredict(x2, YB2, svm1, '-q');
    end;

    % Method: random forrest regression (MATLAB)
    % https://de.mathworks.com/help/stats/select-predictors-for-random-forests.html
    if strcmp(meth{h}, 'RFR')
        rfe1 = fitrensemble(YB1, x1, 'Method', Mrf);
        xp2  = predict(rfe1, YB2);
    end;

    % Method: neural network regression (MATLAB)
    % https://de.mathworks.com/help/stats/classificationneuralnetwork.html
    if strcmp(meth{h}, 'NNR')
        nn1 = fitrnet(YB1, x1, 'LayerSizes', Lnn, 'Activations', Ann);
        xp2 = predict(nn1, YB2);
    end;
    
    % calculate performance
    xp(:,h) = xp2;
    r(h)    = corr(xp(:,h), x2);
    MAE(h)  = mean(abs(xp(:,h)-x2));
    
end;
fprintf('\n\n');
clear mba1 B_est1 s2_est Si_est1 b_est1 svm1 rfe1 nn1


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Analysis 4 (Comparison)', 'Color', [1 1 1], 'Position', [50 50 800 900]);
cols  = 'bcmryw';
x_off = 2;
y_off = 0.95;
hold on;

% plot performance
for h = 1:M
    bar(h, r(h), 0.7, cols(h));
end;
plot([(1-1), (M+2)], [0, 0], ':k', 'LineWidth', 2);
plot([x_off, x_off]+1/2, [0, 1], '-k', 'LineWidth', 1);
axis([(1-1), (M+2), 0, 1]);
set(gca,'Box','On');
set(gca,'XTick',[1:M],'XTickLabel',meth);
legend([meth, {'chance'}], 'Location', 'NorthEast');
xlabel('regression approach', 'FontSize', 16);
ylabel('predictive correlation', 'FontSize', 16);
title('Analysis 4: Comparison', 'FontSize', 24);
text(x_off+1/2, y_off, sprintf('generative   \nmethods   '), ...
     'FontSize', 16, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
text(x_off+1/2, y_off, sprintf('   discriminative\n   methods'), ...
     'FontSize', 16, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
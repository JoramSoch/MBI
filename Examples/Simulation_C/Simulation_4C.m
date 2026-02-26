% Multivariate Bayesian Inversion for Classification and Regression
% Simulation 4: comparison of methods (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/02/2022, 22:38: first MATLAB version
% - 20/02/2025, 16:48: final MATLAB version
% - 26/02/2026, 13:02: added performance plot
% - 26/02/2026, 14:02: rewrote MBR and SVR
% - 26/02/2026, 14:32: added MLR, RFR, NNR
% - 26/02/2026, 15:32: added GNB regression
% - 26/02/2026, 15:44: refined performance plot


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


%%% Step 2: generate data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate data
B = normrnd(mu, sqrt(sb), [size(X,2) v]);
E = matnrnd(zeros(n,v), s2*V, Si, 1);
Y = X*B + E;


%%% Step 3: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify analysis parameters
meth    = {'MBR', 'GNB', 'MLR', 'SVR', 'RFR', 'NNR'};
prior.x = [-xm:0.01:+xm];       % MBR prior distribution
prior.p = (1/range(prior.x))*ones(size(prior.x));
Dgnb    = 'mvn';                % GNB distribution name
Mmlr    = 'WLS';                % MLR estimation method
Csvm    = 1;                    % SVM cost parameter
Mrf     = 'Bag';                % RF aggregation method
Lnn     = [10, 10];             % NN hidden layer sizes
Ann     = 'none';               % NN activation function
CV      = ML_CV(n, k, 'kf');    % k-fold cross-validation

% preallocate results
M   = numel(meth);
xp  = zeros(n,M);
r   = zeros(1,M);
MAE = zeros(1,M);

% perform regression
for g = 1:size(CV,2)

    % get training and test set
    i1 = find(CV(:,g)==1);
    i2 = find(CV(:,g)==2);
    Y1 = Y(i1,:);
    Y2 = Y(i2,:);
    x1 = x(i1);
    x2 = x(i2);
    X1 = X(i1,:);
    X2 = X(i2,:);
    V1 = V(i1,i1);
    V2 = V(i2,i2);

    % evaluate all methods
    for h = 1:M
        
        % Method: multivariate Bayesian regression (MBI)
        % https://github.com/JoramSoch/MBI/blob/main/MATLAB/ML_MBI.m
        if strcmp(meth{h}, 'MBR')
            mba1 = mbitrain(Y1, x1, [], V1, 'MBR');
            pp2  = mbitest(Y2, x2, [], V2, mba1, prior);
            [m,j]= max(pp2, [], 2);
            xp2  = prior.x(j);
        end;
        
        % Method: Gaussian naive Bayes (custom)
        % https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Probabilistic_model
        if strcmp(meth{h}, 'GNB')
            B_est1  = (X1'*X1)^-1 * X1'*Y1;
            s2_est1 = (1/n) * sum((Y1-X1*B_est1).^2);
            Si_est1 = diag(s2_est1);
            log_PP  = zeros(numel(i2),numel(prior.x));
            pp2     = zeros(numel(i2),numel(prior.x));
            for i = 1:numel(i2)
                y2i = Y2(i,:);
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
                b_est1 = (Y1'*Y1)^-1 * Y1'*x1;
            elseif strcmp(Mmlr, 'WLS')
                b_est1 = (Y1'*P1*Y1)^-1 * Y1'*P1*x1;
            else
                b_est1 = zeros(v,1);
            end;
            xp2 = Y2 * b_est1;
        end;
        
        % Method: support vector regression (LibSVM)
        % https://github.com/JoramSoch/ML4ML/blob/main/ML_SVR.m
        if strcmp(meth{h}, 'SVR')
            opt  = sprintf('-s 4 -t 0 -c %s -q', num2str(Csvm));
            svm1 = svmtrain(x1, Y1, opt);
            xp2  = svmpredict(x2, Y2, svm1, '-q');
        end;

        % Method: random forrest regression (MATLAB)
        % https://de.mathworks.com/help/stats/select-predictors-for-random-forests.html
        if strcmp(meth{h}, 'RFR')
            rfe1 = fitrensemble(Y1, x1, 'Method', Mrf);
            xp2  = predict(rfe1, Y2);
        end;

        % Method: neural network regression (MATLAB)
        % https://de.mathworks.com/help/stats/classificationneuralnetwork.html
        if strcmp(meth{h}, 'NNR')
            nn1 = fitrnet(Y1, x1, 'LayerSizes', Lnn, 'Activations', Ann);
            xp2 = predict(nn1, Y2);
        end;
        
        % store test set predictions
        xp(i2,h) = xp2;
        
    end;
    clear mba1 B_est1 s2_est Si_est1 b_est1 svm1 rfe1 nn1
    
end;

% calculate performance
for h = 1:M
    r(h)   = corr(xp(:,h), x);
    MAE(h) = mean(abs(xp(:,h)-x));
end;


%%% Step 4: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Simulation 4 (Comparison)', 'Color', [1 1 1], 'Position', [50 50 800 900]);
cols  = 'bcmryw';
x_off = 2;
y_off = 0.925;
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
title('Simulation 4: Comparison', 'FontSize', 24);
text(x_off+1/2, y_off, sprintf('generative   \nmethods   '), ...
     'FontSize', 16, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
text(x_off+1/2, y_off, sprintf('   discriminative\n   methods'), ...
     'FontSize', 16, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
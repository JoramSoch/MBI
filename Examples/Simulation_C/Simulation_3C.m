% Multivariate Bayesian Inversion for Classification and Regression
% Simulation 3: comparison of methods (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/02/2022, 15:55: first MATLAB version
% - 20/02/2025, 14:57: final MATLAB version
% - 26/02/2026, 10:57: added performance plot
% - 26/02/2026, 11:08: rewrote MBC and SVC
% - 26/02/2026, 11:21: added LDA classification
% - 26/02/2026, 11:32: added GNB classification
% - 26/02/2026, 11:43: added LogReg classification
% - 26/02/2026, 12:01: added RF classification
% - 26/02/2026, 12:15: added NN classification
% - 26/02/2026, 12:44: refined performance plot


clear
close all

%%% Step 1: specify ground truth %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% set ground truth
rng(1);
mu = 1;                         % class means
Si = [1, 0.5; 0.5, 1];          % covariance structure
s2 = 4;                         % noise variance
n  = 300;                       % number of data points
k  = 10;                        % number of CV folds
v  = 2;                         % number of features
C  = 3;                         % number of classes

% generate classes
x  = [kron([1:C]',ones(n/C,1)), rand(n,1)];
x  = sortrows(x,2);
x  = x(:,1);                    % randomized labels
X  = zeros(n,C);                % design matrix
V  = eye(n);                    % observation covariance
for i = 1:n
    X(i,x(i)) = 1;
end;


%%% Step 2: generate data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% generate data
B = [  -mu,   +mu;
     +2*mu, +2*mu;
       +mu,   -mu];
E = matnrnd(zeros(n,v), s2*V, Si, 1);
Y = X*B + E;


%%% Step 3: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify analysis parameters
meth    = {'MBC', 'GNB', 'LDA', 'LogReg', 'SVC', 'RFC', 'NNC'};
prior.x = [1:C];                % MBC class indices
prior.p = 1/C*ones(1,C);        % MBC prior probabilities
Dgnb    = 'normal';             % GNB distribution name
Dlda    = 'linear';             % LDA discriminant type
Llogreg = 'logit';              % LogReg link function
Mlogreg = 'nominal';            % LogReg model type
Csvm    = 1;                    % SVM cost parameter
Mrf     = 'Bag';                % RF aggregation method
Lnn     = [10, 10];             % NN hidden layer sizes
Ann     = 'sigmoid';            % NN activation function
CV      = ML_CV(x, k, 'kfc');   % k-fold CV on points per class

% preallocate results
M  = numel(meth);
xp = zeros(n,M);
CA = zeros(1,M);

% perform classification
for g = 1:size(CV,2)

    % get training and test set
    i1 = find(CV(:,g)==1);
    i2 = find(CV(:,g)==2);
    Y1 = Y(i1,:);
    Y2 = Y(i2,:);
    x1 = x(i1);
    x2 = x(i2);
    V1 = V(i1,i1);
    V2 = V(i2,i2);

    % evaluate all methods
    for h = 1:M
        
        % Method: multivariate Bayesian classification (MBI)
        % https://github.com/JoramSoch/MBI/blob/main/MATLAB/ML_MBI.m
        if strcmp(meth{h}, 'MBC')
            mba1 = mbitrain(Y1, x1, [], V1, 'MBC');
            pp2  = mbitest(Y2, x2, [], V2, mba1, prior);
            [m,j]= max(pp2, [], 2);
            xp2  = prior.x(j);
        end;
        
        % Method: Gaussian naive Bayes (MATLAB)
        % https://de.mathworks.com/help/stats/naive-bayes-classification.html
        if strcmp(meth{h}, 'GNB')
            nbc1 = fitcnb(Y1, x1, 'DistributionNames', Dgnb, 'Prior', prior.p);
            xp2  = predict(nbc1, Y2);
        end;
        
        % Method: linear discriminant analysis (MATLAB)
        % https://de.mathworks.com/help/stats/discriminant-analysis.html
        if strcmp(meth{h}, 'LDA')
            lda1 = fitcdiscr(Y1, x1, 'DiscrimType', Dlda);
            xp2  = predict(lda1, Y2);
        end;
        
        % Method: multinomial logistic regression (MATLAB)
        % https://de.mathworks.com/help/stats/fitmnr.html
        if strcmp(meth{h}, 'LogReg')
            logreg1 = fitmnr(Y1, x1, 'Link', Llogreg, 'ModelType', Mlogreg);
            xp2     = predict(logreg1, Y2);
        end;
        
        % Method: support vector classification (LibSVM)
        % https://github.com/JoramSoch/ML4ML/blob/main/ML_SVC.m
        if strcmp(meth{h}, 'SVC')
            opt  = sprintf('-s 0 -t 0 -c %s -q', num2str(Csvm));
            svm1 = svmtrain(x1, Y1, opt);
            xp2  = svmpredict(x2, Y2, svm1, '-q');
        end;

        % Method: random forrest classification (MATLAB)
        % https://de.mathworks.com/help/stats/select-predictors-for-random-forests.html
        if strcmp(meth{h}, 'RFC')
            rfe1 = fitcensemble(Y1, x1, 'Method', Mrf);
            xp2  = round(predict(rfe1, Y2));
        end;

        % Method: neural network classification (MATLAB)
        % https://de.mathworks.com/help/stats/classificationneuralnetwork.html
        if strcmp(meth{h}, 'NNC')
            nn1 = fitcnet(Y1, x1, 'LayerSizes', Lnn, 'Activations', Ann);
            xp2 = predict(nn1, Y2);
        end;
        
        % store test set predictions
        xp(i2,h) = xp2;
        
    end;
    clear mba1 nbc1 lda1 logreg1 svm1 rfe1 nn1
    
end;

% calculate performance
for h = 1:M
    CA(h) = mean(xp(:,h)==x);
end;


%%% Step 4: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Simulation 3 (Comparison)', 'Color', [1 1 1], 'Position', [50 50 800 900]);
cols  = 'bcgmryw';
x_off = 3;
y_off = 0.9;
hold on;

% plot performance
for h = 1:M
    bar(h, CA(h), 0.7, cols(h));
end;
plot([(1-1), (M+1)], [1/C, 1/C], ':k', 'LineWidth', 2);
plot([x_off, x_off]+1/2, [0, 1], '-k', 'LineWidth', 1);
axis([(1-1), (M+1), 0, 1]);
set(gca,'Box','On');
set(gca,'XTick',[1:M],'XTickLabel',meth);
legend([meth, {'chance'}], 'Location', 'NorthEast');
xlabel('classification approach', 'FontSize', 16);
ylabel('classification accuracy', 'FontSize', 16);
title('Simulation 3: Comparison', 'FontSize', 24);
text(x_off+1/2, y_off, sprintf('generative   \nmethods   '), ...
     'FontSize', 16, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
text(x_off+1/2, y_off, sprintf('   discriminative\n   methods'), ...
     'FontSize', 16, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
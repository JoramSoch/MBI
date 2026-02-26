% Multivariate Bayesian Inversion for Classification and Regression
% Analysis 2: comparison of methods (MATLAB script)
% 
% Author: Joram Soch, OvGU Magdeburg
% E-Mail: joram.soch@ovgu.de
% 
% Version History:
% - 20/12/2024, 17:46: first data analysis
% - 26/02/2025, 16:04: final figure display
% - 26/02/2026, 16:03: rewrote for multiple analyses
% - 26/02/2026, 16:29: added MBC, GNB, LDA, LogReg, SVC, RFC, NNC
% - 26/02/2026, 16:34: restricted to data subsets
% - 26/02/2026, 16:45: remove features for GNB


clear
close all

%%% Step 1: load data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load extracted data
load('../Analysis_2/MNIST_data.mat');

% get data dimensions
n1 = size(Y1,1);                % number of training data points
n2 = size(Y2,1);                % number of test data points
v  = size(Y2,2);                % number of features
C  = max(x2);                   % number of classes


%%% Step 2: analyze data %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% specify analysis parameters
meth    = {'MBC', 'GNB', 'LDA', 'LogReg', 'SVC', 'RFC', 'NNC'};
prior.x = [1:C];                % MBC class indices
prior.p = 1/C*ones(1,C);        % MBC prior probabilities
Dgnb    = 'normal';             % GNB distribution name
Dlda    = 'linear';             % LDA discriminant type
Llogreg = 'logit';              % LogReg link function
Mlogreg = 'ordinal';            % LogReg model type
Ilogreg = false;                % LogReg interactions
Csvm    = 1;                    % SVM cost parameter
Mrf     = 'Bag';                % RF aggregation method
Lnn     = [v];                  % NN hidden layer sizes
Ann     = 'sigmoid';            % NN activation function

% restrict to data subset
N1 = 60000;                     % training data points to use
N2 = 10000;                     % test data points to use

% preallocate results
M  = numel(meth);
xp = zeros(N2,M);
CA = zeros( 1,M);

% display parameters
fprintf('\n-> Analysis 2:\n')
fprintf('   - N1 = %d training data points.\n', N1);
fprintf('   - N2 = %d test data points.\n', N2);

% evaluate all methods
for h = 1:M

    % display method
    fprintf('   - %s: ', meth{h});
    
    % Method: multivariate Bayesian classification (MBI)
    % https://github.com/JoramSoch/MBI/blob/main/MATLAB/ML_MBI.m
    if strcmp(meth{h}, 'MBC')
        fprintf('training ... ')
        mba1 = mbitrain(Y1(1:N1,:), x1(1:N1), [], speye(N1), 'MBC');
        fprintf('testing ... ')
        pp2  = mbitest(Y2(1:N2,:), x2(1:N2), [], speye(N2), mba1, prior);
        [m,j]= max(pp2, [], 2);
        xp2  = prior.x(j);
        fprintf('done.\n')
    end;
    
    % Method: Gaussian naive Bayes (MATLAB)
    % https://de.mathworks.com/help/stats/naive-bayes-classification.html
    if strcmp(meth{h}, 'GNB')
        fprintf('remove at-least-one-class zero-variance features ')
        Y1h = Y1(1:N1,:);
        x1h = x1(1:N1);
        Y1v = zeros(C,v);
        for k = 1:C
            Y1v(k,:) = var(Y1h(x1h==k,:));
        end;
        jh  = all(Y1v);
        fprintf('(%d) \n', v-sum(jh));
        fprintf('          ');
        fprintf('training ... ')
        Y1h  = Y1h(:,jh);
        nbc1 = fitcnb(Y1h, x1h, 'DistributionNames', Dgnb, 'Prior', prior.p);
        fprintf('testing ... ')
        Y2h  = Y2(1:N2,jh);
        xp2  = predict(nbc1, Y2h);
        fprintf('done.\n')
    end;
    
    % Method: linear discriminant analysis (MATLAB)
    % https://de.mathworks.com/help/stats/discriminant-analysis.html
    if strcmp(meth{h}, 'LDA')
        fprintf('training ... ')
        lda1 = fitcdiscr(Y1(1:N1,:), x1(1:N1), 'DiscrimType', Dlda);
        fprintf('testing ... ')
        xp2  = predict(lda1, Y2(1:N2,:));
        fprintf('done.\n')
    end;
    
    % Method: multinomial logistic regression (MATLAB)
    % https://de.mathworks.com/help/stats/fitmnr.html
    if strcmp(meth{h}, 'LogReg')
        fprintf('training ... ')
        logreg1 = fitmnr(Y1(1:N1,:), x1(1:N1), 'Link', Llogreg, 'ModelType', Mlogreg, ...
                                               'IncludeClassInteractions', Ilogreg);
        fprintf('testing ... ')
        xp2     = predict(logreg1, Y2(1:N2,:));
        fprintf('done.\n')
    end;
    
    % Method: support vector classification (LibSVM)
    % https://github.com/JoramSoch/ML4ML/blob/main/ML_SVC.m
    if strcmp(meth{h}, 'SVC')
        fprintf('training ... ')
        opt  = sprintf('-s 0 -t 0 -c %s -q', num2str(Csvm));
        svm1 = svmtrain(x1(1:N1), Y1(1:N1,:), opt);
        fprintf('testing ... ')
        xp2  = svmpredict(x2(1:N2), Y2(1:N2,:), svm1, '-q');
        fprintf('done.\n')
    end;

    % Method: random forrest classification (MATLAB)
    % https://de.mathworks.com/help/stats/select-predictors-for-random-forests.html
    if strcmp(meth{h}, 'RFC')
        fprintf('training ... ')
        rfe1 = fitrensemble(Y1(1:N1,:), x1(1:N1), 'Method', Mrf);
        fprintf('testing ... ')
        xp2  = round(predict(rfe1, Y2(1:N2,:)));
        fprintf('done.\n')
    end;

    % Method: neural network classification (MATLAB)
    % https://de.mathworks.com/help/stats/classificationneuralnetwork.html
    if strcmp(meth{h}, 'NNC')
        fprintf('training ... ')
        nn1 = fitcnet(Y1(1:N1,:), x1(1:N1), 'LayerSizes', Lnn, 'Activations', Ann);
        fprintf('testing ... ')
        xp2 = predict(nn1, Y2(1:N2,:));
        fprintf('done.\n')
    end;
    
    % calculate performance
    xp(:,h) = xp2;
    CA(h)   = mean(xp(:,h)==x2(1:N2));

end;
fprintf('\n\n');
clear mba1 nbc1 lda1 logreg1 svm1 rfe1 nn1


%%% Step 3: visualize results %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% open figure
figure('Name', 'Analysis 2 (Comparison)', 'Color', [1 1 1], 'Position', [50 50 800 900]);
cols  = 'bcgmryw';
x_off = 3;
y_off = 0.95;
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
legend([meth, {'chance'}], 'Location', 'Best');
xlabel('classification approach', 'FontSize', 16);
ylabel('classification accuracy', 'FontSize', 16);
title('Analysis 2: Comparison', 'FontSize', 24);
text(x_off+1/2, y_off, sprintf('generative   \nmethods   '), ...
     'FontSize', 16, 'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
text(x_off+1/2, y_off, sprintf('   discriminative\n   methods'), ...
     'FontSize', 16, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle');
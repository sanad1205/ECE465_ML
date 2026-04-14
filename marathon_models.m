%% ============================================================
%  Predicting Marathon Race Performance Using Machine Learning
%  ------------------------------------------------------------
%  Project Progress Update #2
%
%  Course: IECE 465 - Introduction to Machine Learning for Engineers
%  Faculty Mentor: Dr. Hany Elgala
%  Teaching Assistant: Honglan Chen
%
%  Team Members:
%   - Sanad Sahawneh
%   - Brian Allen
%   - Mahib Rahman
%
%  Dates:
%   - Monday, April 13th, 2026
%   - Wednesday, April 15th, 2026
%
%  ------------------------------------------------------------
%  Dataset:
%   - Boston Marathon 2023 dataset
%
%  Project Overview:
%  This project focuses on predicting marathon finish times using
%  machine learning techniques. The goal is to model the relationship
%  between runner characteristics (age group, gender) and performance
%  metrics (half-marathon split time) to estimate final finish time.
%
%  ------------------------------------------------------------
%  Features:
%   - age_group
%   - gender
%   - half_time_sec
%
%  Target:
%   - finish_net_sec
%
%  ------------------------------------------------------------
%  Models Implemented:
%   - Baseline Predictor (ratio-based)
%   - Linear Regression
%   - Neural Network Regression
%
%  ------------------------------------------------------------
%  Evaluation Metrics:
%   - Mean Absolute Error (MAE)
%   - Root Mean Squared Error (RMSE)
%   - R-squared (R^2)
%   - Mean Absolute Percentage Error (MAPE)
%   - Error interpretation in minutes
%
%  ------------------------------------------------------------
%  Prediction Quality Interpretation:
%   - Excellent: average error under 5 minutes
%   - Strong: average error between 5 and 10 minutes
%   - Reasonable: average error between 10 and 15 minutes
%   - Weak: average error above 15 minutes
%
%  Sports Betting Style Extension:
%   - Uses predicted finish time to evaluate hypothetical
%     over/under time lines
%   - Measures whether the runner finishes over or under
%     a selected benchmark time
%   - Computes accuracy of threshold-based outcome prediction
%
%  ------------------------------------------------------------
%  Visualizations:
%   - Actual vs Predicted plots
%   - Residual plots
%   - Error distribution histograms
%   - Model comparison bar charts (MAE, RMSE, R^2, MAPE)
%   - Feature relationship analysis (age, gender, split time)
%   - Linear model coefficient visualization
%   - Sports betting style over/under threshold analysis plots
%
% ============================================================

clear; clc; close all;

%% -----------------------------
% Load and prepare dataset
% -----------------------------
data = readtable('boston_marathon_2023.csv');

% Keep only the columns needed
data = data(:, {'age_group','gender','half_time_sec','finish_net_sec'});

% Remove missing rows
data = rmmissing(data);

% Remove any invalid or zero times if they exist
data = data(data.half_time_sec > 0 & data.finish_net_sec > 0, :);

% Convert to categorical
data.age_group = categorical(data.age_group);
data.gender    = categorical(data.gender);

disp(' ')
disp('================ DATA SUMMARY ================')
disp(['Total usable samples: ' num2str(height(data))])
summary(data)

%% -----------------------------
% Train/Test Split
% -----------------------------
rng(42); % reproducibility
cv = cvpartition(height(data), 'HoldOut', 0.20);

trainData = data(training(cv), :);
testData  = data(test(cv), :);

yTrain = trainData.finish_net_sec;
yTest  = testData.finish_net_sec;

disp(' ')
disp('================ SPLIT SUMMARY ================')
disp(['Training samples: ' num2str(height(trainData))])
disp(['Testing samples:  ' num2str(height(testData))])

%% -----------------------------
% BASELINE MODEL
% Predict finish time directly from average ratio in training set:
% finish ≈ half_time * average(train finish / train half)
% -----------------------------
avgRatio = mean(trainData.finish_net_sec ./ trainData.half_time_sec);
yPred_base = testData.half_time_sec .* avgRatio;

%% -----------------------------
% MODEL 1: Linear Regression
% -----------------------------
linModel = fitlm(trainData, 'finish_net_sec ~ age_group + gender + half_time_sec');
yPred_lin = predict(linModel, testData);

%% -----------------------------
% MODEL 2: Neural Network Regression
% -----------------------------
% Ensure categorical consistency between train and test
ageTrain = categorical(trainData.age_group, categories(data.age_group));
ageTest  = categorical(testData.age_group,  categories(data.age_group));

genderTrain = categorical(trainData.gender, categories(data.gender));
genderTest  = categorical(testData.gender,  categories(data.gender));

% One-hot encoding
Xtrain_age    = dummyvar(ageTrain);
Xtest_age     = dummyvar(ageTest);
Xtrain_gender = dummyvar(genderTrain);
Xtest_gender  = dummyvar(genderTest);

% Numeric feature
Xtrain_num = trainData.half_time_sec;
Xtest_num  = testData.half_time_sec;

% Combine features
Xtrain = [Xtrain_age Xtrain_gender Xtrain_num];
Xtest  = [Xtest_age  Xtest_gender  Xtest_num];

% Standardize inputs using training data only
mu = mean(Xtrain, 1);
sigma = std(Xtrain, 0, 1);
sigma(sigma == 0) = 1;

Xtrain_std = (Xtrain - mu) ./ sigma;
Xtest_std  = (Xtest  - mu) ./ sigma;

% Neural network
nnModel = fitrnet(Xtrain_std, yTrain, ...
    'LayerSizes', [64 32 16], ...
    'Activations', 'relu', ...
    'Standardize', false);

yPred_nn = predict(nnModel, Xtest_std);

%% -----------------------------
% Metric function
% -----------------------------
calcMetrics = @(yTrue, yPred) struct( ...
    'MAE', mean(abs(yTrue - yPred)), ...
    'RMSE', sqrt(mean((yTrue - yPred).^2)), ...
    'MAPE', mean(abs((yTrue - yPred) ./ yTrue)) * 100, ...
    'R2', 1 - sum((yTrue - yPred).^2) / sum((yTrue - mean(yTrue)).^2), ...
    'MAE_min', mean(abs(yTrue - yPred)) / 60, ...
    'RMSE_min', sqrt(mean((yTrue - yPred).^2)) / 60);

metrics_base = calcMetrics(yTest, yPred_base);
metrics_lin  = calcMetrics(yTest, yPred_lin);
metrics_nn   = calcMetrics(yTest, yPred_nn);

%% -----------------------------
% Prediction quality interpretation
% -----------------------------
if metrics_base.MAE_min < 5
    quality_base = "Excellent (< 5 min average error)";
elseif metrics_base.MAE_min < 10
    quality_base = "Strong (5-10 min average error)";
elseif metrics_base.MAE_min < 15
    quality_base = "Reasonable (10-15 min average error)";
else
    quality_base = "Weak (> 15 min average error)";
end

if metrics_lin.MAE_min < 5
    quality_lin = "Excellent (< 5 min average error)";
elseif metrics_lin.MAE_min < 10
    quality_lin = "Strong (5-10 min average error)";
elseif metrics_lin.MAE_min < 15
    quality_lin = "Reasonable (10-15 min average error)";
else
    quality_lin = "Weak (> 15 min average error)";
end

if metrics_nn.MAE_min < 5
    quality_nn = "Excellent (< 5 min average error)";
elseif metrics_nn.MAE_min < 10
    quality_nn = "Strong (5-10 min average error)";
elseif metrics_nn.MAE_min < 15
    quality_nn = "Reasonable (10-15 min average error)";
else
    quality_nn = "Weak (> 15 min average error)";
end

%% -----------------------------
% Display results
% -----------------------------
disp(' ')
disp('================ MODEL RESULTS ================')

Model = ["Baseline"; "Linear Regression"; "Neural Network"];
MAE_sec  = [metrics_base.MAE;  metrics_lin.MAE;  metrics_nn.MAE];
RMSE_sec = [metrics_base.RMSE; metrics_lin.RMSE; metrics_nn.RMSE];
MAE_min  = [metrics_base.MAE_min;  metrics_lin.MAE_min;  metrics_nn.MAE_min];
RMSE_min = [metrics_base.RMSE_min; metrics_lin.RMSE_min; metrics_nn.RMSE_min];
MAPE_pct = [metrics_base.MAPE; metrics_lin.MAPE; metrics_nn.MAPE];
R2_score = [metrics_base.R2; metrics_lin.R2; metrics_nn.R2];

resultsTable = table(Model, MAE_sec, RMSE_sec, MAE_min, RMSE_min, MAPE_pct, R2_score);
disp(resultsTable)

disp(' ')
disp('================ PREDICTION QUALITY ================')
disp(['Baseline:          ' char(quality_base)])
disp(['Linear Regression: ' char(quality_lin)])
disp(['Neural Network:    ' char(quality_nn)])

%% -----------------------------
% Choose best model based on RMSE
% -----------------------------
[~, bestIdx] = min(RMSE_sec);
bestModelName = Model(bestIdx);

disp(' ')
disp('================ BEST MODEL ================')
disp(['Best model by RMSE: ' char(bestModelName)])

%% -----------------------------
% Select best prediction model
% -----------------------------
if bestIdx == 1
    yPred_best = yPred_base;
elseif bestIdx == 2
    yPred_best = yPred_lin;
else
    yPred_best = yPred_nn;
end

%% -----------------------------
% Sports betting style extension:
% Multi-line over/under analysis
%
% This does not use sportsbook odds. Instead, it converts the
% finish-time prediction problem into a threshold-based decision task:
% will a runner finish OVER or UNDER a benchmark time line?
% -----------------------------
betLines_min = [180 210 240 270 300]; % 3:00, 3:30, 4:00, 4:30, 5:00

betAccuracies   = zeros(length(betLines_min),1);
underRates_actual = zeros(length(betLines_min),1);
underRates_pred   = zeros(length(betLines_min),1);
correctUnderPicks = zeros(length(betLines_min),1);
correctOverPicks  = zeros(length(betLines_min),1);
falseUnderPicks   = zeros(length(betLines_min),1);
falseOverPicks    = zeros(length(betLines_min),1);

for i = 1:length(betLines_min)
    line_sec = betLines_min(i) * 60;

    actualUnder = yTest < line_sec;
    predUnder   = yPred_best < line_sec;

    betAccuracies(i) = mean(actualUnder == predUnder) * 100;
    underRates_actual(i) = mean(actualUnder) * 100;
    underRates_pred(i)   = mean(predUnder) * 100;

    correctUnderPicks(i) = sum(actualUnder == 1 & predUnder == 1);
    correctOverPicks(i)  = sum(actualUnder == 0 & predUnder == 0);
    falseUnderPicks(i)   = sum(actualUnder == 0 & predUnder == 1);
    falseOverPicks(i)    = sum(actualUnder == 1 & predUnder == 0);
end

betTable = table(betLines_min', betAccuracies, underRates_actual, underRates_pred, ...
    correctUnderPicks, correctOverPicks, falseUnderPicks, falseOverPicks, ...
    'VariableNames', {'Line_Minutes','ThresholdAccuracy_pct','ActualUnder_pct', ...
    'PredictedUnder_pct','CorrectUnderPicks','CorrectOverPicks', ...
    'FalseUnderPicks','FalseOverPicks'});

disp(' ')
disp('================ SPORTS BETTING STYLE ANALYSIS ================')
disp(betTable)

%% -----------------------------
% Residuals
% -----------------------------
res_base = yTest - yPred_base;
res_lin  = yTest - yPred_lin;
res_nn   = yTest - yPred_nn;

%% ============================================================
% VISUALS
% ============================================================

%% -----------------------------
% Plot 1: Actual vs Predicted - Baseline
% -----------------------------
figure;
scatter(yTest/60, yPred_base/60, 25, 'filled')
hold on
minVal = min([yTest; yPred_base])/60;
maxVal = max([yTest; yPred_base])/60;
plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.5)
xlabel('Actual Finish Time (minutes)')
ylabel('Predicted Finish Time (minutes)')
title('Baseline Model: Actual vs Predicted')
grid on
axis equal
hold off

%% -----------------------------
% Plot 2: Actual vs Predicted - Linear Regression
% -----------------------------
figure;
scatter(yTest/60, yPred_lin/60, 25, 'filled')
hold on
minVal = min([yTest; yPred_lin])/60;
maxVal = max([yTest; yPred_lin])/60;
plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.5)
xlabel('Actual Finish Time (minutes)')
ylabel('Predicted Finish Time (minutes)')
title('Linear Regression: Actual vs Predicted')
grid on
axis equal
hold off

%% -----------------------------
% Plot 3: Actual vs Predicted - Neural Network
% -----------------------------
figure;
scatter(yTest/60, yPred_nn/60, 25, 'filled')
hold on
minVal = min([yTest; yPred_nn])/60;
maxVal = max([yTest; yPred_nn])/60;
plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.5)
xlabel('Actual Finish Time (minutes)')
ylabel('Predicted Finish Time (minutes)')
title('Neural Network: Actual vs Predicted')
grid on
axis equal
hold off

%% -----------------------------
% Plot 4: Residuals vs Predicted - Linear Regression
% -----------------------------
figure;
scatter(yPred_lin/60, res_lin/60, 25, 'filled')
hold on
yline(0, 'r--', 'LineWidth', 1.5)
xlabel('Predicted Finish Time (minutes)')
ylabel('Residual Error (minutes)')
title('Linear Regression Residuals')
grid on
hold off

%% -----------------------------
% Plot 5: Residuals vs Predicted - Neural Network
% -----------------------------
figure;
scatter(yPred_nn/60, res_nn/60, 25, 'filled')
hold on
yline(0, 'r--', 'LineWidth', 1.5)
xlabel('Predicted Finish Time (minutes)')
ylabel('Residual Error (minutes)')
title('Neural Network Residuals')
grid on
hold off

%% -----------------------------
% Plot 6: Error Histogram Comparison
% -----------------------------
figure;
histogram(res_lin/60, 30, 'Normalization', 'probability')
hold on
histogram(res_nn/60, 30, 'Normalization', 'probability')
xlabel('Prediction Error (minutes)')
ylabel('Probability')
title('Error Distribution: Linear Regression vs Neural Network')
legend('Linear Regression', 'Neural Network', 'Location', 'best')
grid on
hold off

%% -----------------------------
% Plot 7: MAE Comparison
% -----------------------------
figure;
bar(MAE_min)
set(gca, 'XTickLabel', Model)
ylabel('MAE (minutes)')
title('Model Comparison: MAE')
grid on

%% -----------------------------
% Plot 8: RMSE Comparison
% -----------------------------
figure;
bar(RMSE_min)
set(gca, 'XTickLabel', Model)
ylabel('RMSE (minutes)')
title('Model Comparison: RMSE')
grid on

%% -----------------------------
% Plot 9: R^2 Comparison
% -----------------------------
figure;
bar(R2_score)
set(gca, 'XTickLabel', Model)
ylabel('R^2')
title('Model Comparison: R^2')
grid on

%% -----------------------------
% Plot 10: MAPE Comparison
% -----------------------------
figure;
bar(MAPE_pct)
set(gca, 'XTickLabel', Model)
ylabel('MAPE (%)')
title('Model Comparison: MAPE')
grid on

%% -----------------------------
% Plot 11: Half Split vs Finish Time
% -----------------------------
figure;
scatter(data.half_time_sec/60, data.finish_net_sec/60, 15, 'filled')
xlabel('Half Marathon Split (minutes)')
ylabel('Finish Time (minutes)')
title('Half Split Time vs Finish Time')
grid on

%% -----------------------------
% Plot 12: Average finish time by age group
% -----------------------------
ageCats = categories(data.age_group);
meanFinishByAge = zeros(length(ageCats), 1);

for i = 1:length(ageCats)
    idx = data.age_group == ageCats{i};
    meanFinishByAge(i) = mean(data.finish_net_sec(idx))/60;
end

figure;
bar(meanFinishByAge)
set(gca, 'XTick', 1:length(ageCats), 'XTickLabel', ageCats)
xtickangle(45)
ylabel('Average Finish Time (minutes)')
title('Average Finish Time by Age Group')
grid on

%% -----------------------------
% Plot 13: Average finish time by gender
% -----------------------------
genderCats = categories(data.gender);
meanFinishByGender = zeros(length(genderCats), 1);

for i = 1:length(genderCats)
    idx = data.gender == genderCats{i};
    meanFinishByGender(i) = mean(data.finish_net_sec(idx))/60;
end

figure;
bar(meanFinishByGender)
set(gca, 'XTick', 1:length(genderCats), 'XTickLabel', genderCats)
ylabel('Average Finish Time (minutes)')
title('Average Finish Time by Gender')
grid on

%% -----------------------------
% Plot 14: Linear model coefficients
% -----------------------------
figure;
coefTable = linModel.Coefficients;
coefNames = string(coefTable.Properties.RowNames);
coefVals = coefTable.Estimate;

bar(coefVals)
set(gca, 'XTick', 1:length(coefVals), 'XTickLabel', coefNames)
xtickangle(45)
ylabel('Coefficient Value')
title('Linear Regression Coefficients')
grid on

%% -----------------------------
% Plot 15: Sports betting style over/under accuracy by line
% -----------------------------
figure;
bar(betLines_min, betAccuracies)
xlabel('Betting Line (minutes)')
ylabel('Threshold Prediction Accuracy (%)')
title('Sports Betting Style Analysis: Over/Under Accuracy')
grid on

%% -----------------------------
% Plot 16: Actual vs Predicted Under Rates by line
% -----------------------------
figure;
plot(betLines_min, underRates_actual, '-o', 'LineWidth', 1.5)
hold on
plot(betLines_min, underRates_pred, '-s', 'LineWidth', 1.5)
xlabel('Betting Line (minutes)')
ylabel('Under Rate (%)')
title('Actual vs Predicted Under Rates Across Threshold Lines')
legend('Actual Under Rate', 'Predicted Under Rate', 'Location', 'best')
grid on
hold off

%% -----------------------------
% Plot 17: Best model actual vs predicted
% -----------------------------
figure;
scatter(yTest/60, yPred_best/60, 25, 'filled')
hold on
minVal = min([yTest; yPred_best])/60;
maxVal = max([yTest; yPred_best])/60;
plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.5)
xlabel('Actual Finish Time (minutes)')
ylabel('Predicted Finish Time (minutes)')
title(['Best Model (' char(bestModelName) '): Actual vs Predicted'])
grid on
axis equal
hold off

%% -----------------------------
% Interpretation printout
% -----------------------------
disp(' ')
disp('================ INTERPRETATION NOTES ================')
disp(['Baseline MAE      (min): ' num2str(metrics_base.MAE_min)])
disp(['Linear MAE        (min): ' num2str(metrics_lin.MAE_min)])
disp(['Neural Network MAE(min): ' num2str(metrics_nn.MAE_min)])
disp(' ')
disp('Prediction-quality guide:')
disp(' - Under 5 minutes   : Excellent')
disp(' - 5 to 10 minutes   : Strong')
disp(' - 10 to 15 minutes  : Reasonable')
disp(' - Above 15 minutes  : Weak')
disp(' ')
disp('Lower MAE and RMSE are better.')
disp('MAE tells the average prediction error.')
disp('RMSE penalizes larger mistakes more heavily.')
disp('R^2 closer to 1 means the model explains more variance.')
disp('MAPE gives the average percent error.')
disp(' ')
disp('Sports betting style extension:')
disp('This converts marathon finish-time regression into an over/under')
disp('threshold prediction task using hypothetical benchmark time lines.')
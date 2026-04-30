%% ============================================================
%  Predicting Marathon Race Performance Using Machine Learning
%  ------------------------------------------------------------
%  FINAL PROJECT PRESENTATION VERSION
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
%  Final Presentation Focus:
%   - Latest performance compared to Project Progress Update #2
%   - Completed end-to-end ML system
%   - Technical obstacles and solutions
%   - Lessons learned
%   - Individual team member contributions
%   - Quick demo prediction for a sample runner
%
%  Dataset:
%   - Boston Marathon 2023 dataset
%
%  Features:
%   - age_group
%   - gender
%   - half_time_sec
%
%  Target:
%   - finish_net_sec
%
%  Models:
%   - Baseline ratio predictor
%   - Linear regression
%   - Neural network regression
%   - Ensemble regression, if available
%
%  Outputs Saved:
%   - final_outputs/final_model_results.csv
%   - final_outputs/final_predictions.csv
%   - final_outputs/final_betting_threshold_results.csv
%   - final_outputs/*.png figures
%   - final_outputs/final_best_model.mat
%
% ============================================================

clear; clc; close all;

%% ============================================================
% 0. Project Setup
% ============================================================

rng(42); % reproducibility

dataFile = 'boston_marathon_2023.csv';
outputDir = 'final_outputs';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

fprintf('\n============================================================\n');
fprintf(' FINAL PROJECT: MARATHON RACE PERFORMANCE ML SYSTEM\n');
fprintf('============================================================\n');

%% ============================================================
% 1. Load and Prepare Dataset
% ============================================================

if ~isfile(dataFile)
    error('Dataset file "%s" was not found. Place it in the same folder as this script.', dataFile);
end

data = readtable(dataFile);

requiredColumns = {'age_group','gender','half_time_sec','finish_net_sec'};
missingColumns = setdiff(requiredColumns, data.Properties.VariableNames);

if ~isempty(missingColumns)
    error('Missing required column(s): %s', strjoin(missingColumns, ', '));
end

% Keep only required columns
data = data(:, requiredColumns);

% Remove missing rows
data = rmmissing(data);

% Remove invalid race times
data = data(data.half_time_sec > 0 & data.finish_net_sec > 0, :);

% Remove unrealistic pacing outliers using finish/half ratio
% A reasonable marathon finish time is usually a little more than 2x half split.
data.finish_to_half_ratio = data.finish_net_sec ./ data.half_time_sec;
data = data(data.finish_to_half_ratio >= 1.80 & data.finish_to_half_ratio <= 3.20, :);
data.finish_to_half_ratio = [];

% Convert categorical features
data.age_group = categorical(data.age_group);
data.gender    = categorical(data.gender);

fprintf('\n================ DATA SUMMARY ================\n');
fprintf('Total usable samples after cleaning: %d\n', height(data));
disp(summary(data));

%% ============================================================
% 2. Train/Test Split
% ============================================================

cv = cvpartition(height(data), 'HoldOut', 0.20);

trainData = data(training(cv), :);
testData  = data(test(cv), :);

yTrain = trainData.finish_net_sec;
yTest  = testData.finish_net_sec;

fprintf('\n================ SPLIT SUMMARY ================\n');
fprintf('Training samples: %d\n', height(trainData));
fprintf('Testing samples:  %d\n', height(testData));

%% ============================================================
% 3. Feature Engineering for Numeric ML Models
% ============================================================

[Xtrain, Xtest, featureNames, normParams] = buildFeatureMatrix(trainData, testData, data);

%% ============================================================
% 4. Baseline Model
% ============================================================
% Baseline:
% finish_time ≈ half_time × average training finish/half ratio

avgRatio = mean(trainData.finish_net_sec ./ trainData.half_time_sec);
yPred_base = testData.half_time_sec .* avgRatio;

models = {};
predictions = {};
modelObjects = {};

models{end+1} = 'Baseline Ratio';
predictions{end+1} = yPred_base;
modelObjects{end+1} = struct('Type','Baseline','AverageRatio',avgRatio);

%% ============================================================
% 5. Linear Regression Model
% ============================================================

linModel = fitlm(trainData, 'finish_net_sec ~ age_group + gender + half_time_sec');
yPred_lin = predict(linModel, testData);

models{end+1} = 'Linear Regression';
predictions{end+1} = yPred_lin;
modelObjects{end+1} = linModel;

%% ============================================================
% 6. Neural Network Regression Model
% ============================================================

try
    nnModel = fitrnet(Xtrain, yTrain, ...
        'LayerSizes', [64 32 16], ...
        'Activations', 'relu', ...
        'Standardize', false, ...
        'IterationLimit', 1000);

    yPred_nn = predict(nnModel, Xtest);

    models{end+1} = 'Neural Network';
    predictions{end+1} = yPred_nn;
    modelObjects{end+1} = nnModel;

catch ME
    warning('Neural network model skipped: %s', ME.message);
end

%% ============================================================
% 7. Ensemble Regression Model
% ============================================================

try
    ensModel = fitrensemble(Xtrain, yTrain, ...
        'Method', 'Bag', ...
        'NumLearningCycles', 150, ...
        'Learners', templateTree('MinLeafSize', 10));

    yPred_ens = predict(ensModel, Xtest);

    models{end+1} = 'Bagged Trees Ensemble';
    predictions{end+1} = yPred_ens;
    modelObjects{end+1} = ensModel;

catch ME
    warning('Ensemble model skipped: %s', ME.message);
end

%% ============================================================
% 8. Evaluate All Models
% ============================================================

numModels = numel(models);

MAE_sec  = zeros(numModels,1);
RMSE_sec = zeros(numModels,1);
MAE_min  = zeros(numModels,1);
RMSE_min = zeros(numModels,1);
MAPE_pct = zeros(numModels,1);
R2_score = zeros(numModels,1);
MedianAE_min = zeros(numModels,1);
Within5min_pct = zeros(numModels,1);
Within10min_pct = zeros(numModels,1);
Within15min_pct = zeros(numModels,1);

for i = 1:numModels
    metrics = calcMetrics(yTest, predictions{i});
    MAE_sec(i) = metrics.MAE;
    RMSE_sec(i) = metrics.RMSE;
    MAE_min(i) = metrics.MAE_min;
    RMSE_min(i) = metrics.RMSE_min;
    MAPE_pct(i) = metrics.MAPE;
    R2_score(i) = metrics.R2;
    MedianAE_min(i) = metrics.MedianAE_min;
    Within5min_pct(i) = metrics.Within5min_pct;
    Within10min_pct(i) = metrics.Within10min_pct;
    Within15min_pct(i) = metrics.Within15min_pct;
end

Model = string(models(:));

resultsTable = table(Model, MAE_sec, RMSE_sec, MAE_min, RMSE_min, ...
    MedianAE_min, MAPE_pct, R2_score, Within5min_pct, Within10min_pct, Within15min_pct);

resultsTable = sortrows(resultsTable, 'RMSE_min', 'ascend');

fprintf('\n================ FINAL MODEL RESULTS ================\n');
disp(resultsTable);

writetable(resultsTable, fullfile(outputDir, 'final_model_results.csv'));

%% ============================================================
% 9. Select Best Model
% ============================================================

bestModelName = resultsTable.Model(1);
bestIdx = find(Model == bestModelName, 1);

yPred_best = predictions{bestIdx};
bestModelObject = modelObjects{bestIdx};

fprintf('\n================ BEST FINAL MODEL ================\n');
fprintf('Best model by RMSE: %s\n', bestModelName);
fprintf('MAE:  %.2f minutes\n', resultsTable.MAE_min(1));
fprintf('RMSE: %.2f minutes\n', resultsTable.RMSE_min(1));
fprintf('R^2:  %.4f\n', resultsTable.R2_score(1));
fprintf('Predictions within 10 minutes: %.2f%%\n', resultsTable.Within10min_pct(1));

%% ============================================================
% 10. Final Prediction Table
% ============================================================

predictionTable = testData;
predictionTable.actual_finish_min = yTest / 60;
predictionTable.predicted_finish_min = yPred_best / 60;
predictionTable.error_min = (yTest - yPred_best) / 60;
predictionTable.absolute_error_min = abs(predictionTable.error_min);

writetable(predictionTable, fullfile(outputDir, 'final_predictions.csv'));

%% ============================================================
% 11. Sports Betting Style Over/Under Extension
% ============================================================
% Converts regression into a threshold classification task:
% Will the runner finish UNDER or OVER a selected benchmark time?

betLines_min = [180 210 240 270 300]; % 3:00, 3:30, 4:00, 4:30, 5:00

betAccuracies = zeros(length(betLines_min),1);
actualUnderRates = zeros(length(betLines_min),1);
predictedUnderRates = zeros(length(betLines_min),1);

for i = 1:length(betLines_min)
    line_sec = betLines_min(i) * 60;

    actualUnder = yTest < line_sec;
    predictedUnder = yPred_best < line_sec;

    betAccuracies(i) = mean(actualUnder == predictedUnder) * 100;
    actualUnderRates(i) = mean(actualUnder) * 100;
    predictedUnderRates(i) = mean(predictedUnder) * 100;
end

betTable = table(betLines_min', betAccuracies, actualUnderRates, predictedUnderRates, ...
    'VariableNames', {'Line_Minutes','ThresholdAccuracy_pct','ActualUnder_pct','PredictedUnder_pct'});

fprintf('\n================ OVER/UNDER THRESHOLD DEMO ================\n');
disp(betTable);

writetable(betTable, fullfile(outputDir, 'final_betting_threshold_results.csv'));

%% ============================================================
% 12. Demo Prediction for Final Presentation
% ============================================================

demoRunner.age_group = string(testData.age_group(1));
demoRunner.gender = string(testData.gender(1));
demoRunner.half_time_sec = testData.half_time_sec(1);
demoActual_sec = testData.finish_net_sec(1);

demoPred_sec = predictSingleRunner(bestModelName, bestModelObject, demoRunner, ...
    avgRatio, featureNames, normParams, data);

fprintf('\n================ QUICK DEMO PREDICTION ================\n');
fprintf('Sample Runner:\n');
fprintf('  Age Group: %s\n', demoRunner.age_group);
fprintf('  Gender: %s\n', demoRunner.gender);
fprintf('  Half Marathon Split: %.2f minutes\n', demoRunner.half_time_sec/60);
fprintf('Prediction using best model (%s):\n', bestModelName);
fprintf('  Predicted Finish Time: %.2f minutes\n', demoPred_sec/60);
fprintf('  Actual Finish Time:    %.2f minutes\n', demoActual_sec/60);
fprintf('  Absolute Error:        %.2f minutes\n', abs(demoActual_sec-demoPred_sec)/60);

%% ============================================================
% 13. Save Best Model Package
% ============================================================

bestModelPackage = struct();
bestModelPackage.bestModelName = bestModelName;
bestModelPackage.bestModelObject = bestModelObject;
bestModelPackage.avgRatio = avgRatio;
bestModelPackage.featureNames = featureNames;
bestModelPackage.normParams = normParams;
bestModelPackage.resultsTable = resultsTable;
bestModelPackage.trainingCategories.age_group = categories(data.age_group);
bestModelPackage.trainingCategories.gender = categories(data.gender);

save(fullfile(outputDir, 'final_best_model.mat'), 'bestModelPackage');

%% ============================================================
% 14. Visualizations for Final Presentation
% ============================================================

% Actual vs predicted for best model
fig = figure;
scatter(yTest/60, yPred_best/60, 22, 'filled');
hold on;
minVal = min([yTest; yPred_best]) / 60;
maxVal = max([yTest; yPred_best]) / 60;
plot([minVal maxVal], [minVal maxVal], 'r--', 'LineWidth', 1.5);
xlabel('Actual Finish Time (minutes)');
ylabel('Predicted Finish Time (minutes)');
title(['Best Model Actual vs Predicted: ' char(bestModelName)]);
grid on;
axis equal;
hold off;
saveFigure(fig, outputDir, '01_best_model_actual_vs_predicted.png');

% Residual plot
fig = figure;
res_best = (yTest - yPred_best) / 60;
scatter(yPred_best/60, res_best, 22, 'filled');
hold on;
yline(0, 'r--', 'LineWidth', 1.5);
xlabel('Predicted Finish Time (minutes)');
ylabel('Residual Error (minutes)');
title(['Residuals for Best Model: ' char(bestModelName)]);
grid on;
hold off;
saveFigure(fig, outputDir, '02_best_model_residuals.png');

% Error distribution
fig = figure;
histogram(abs(res_best), 30);
xlabel('Absolute Prediction Error (minutes)');
ylabel('Number of Runners');
title(['Absolute Error Distribution: ' char(bestModelName)]);
grid on;
saveFigure(fig, outputDir, '03_absolute_error_distribution.png');

% MAE comparison
fig = figure;
bar(categorical(resultsTable.Model), resultsTable.MAE_min);
ylabel('MAE (minutes)');
title('Final Model Comparison: Mean Absolute Error');
grid on;
saveFigure(fig, outputDir, '04_model_comparison_mae.png');

% RMSE comparison
fig = figure;
bar(categorical(resultsTable.Model), resultsTable.RMSE_min);
ylabel('RMSE (minutes)');
title('Final Model Comparison: Root Mean Squared Error');
grid on;
saveFigure(fig, outputDir, '05_model_comparison_rmse.png');

% R2 comparison
fig = figure;
bar(categorical(resultsTable.Model), resultsTable.R2_score);
ylabel('R^2 Score');
title('Final Model Comparison: R^2');
grid on;
saveFigure(fig, outputDir, '06_model_comparison_r2.png');

% Half split vs finish time
fig = figure;
scatter(data.half_time_sec/60, data.finish_net_sec/60, 12, 'filled');
xlabel('Half Marathon Split (minutes)');
ylabel('Finish Time (minutes)');
title('Feature Relationship: Half Split vs Finish Time');
grid on;
saveFigure(fig, outputDir, '07_half_split_vs_finish.png');

% Average finish time by age group
ageCats = categories(data.age_group);
meanFinishByAge = zeros(length(ageCats), 1);

for i = 1:length(ageCats)
    idx = data.age_group == ageCats{i};
    meanFinishByAge(i) = mean(data.finish_net_sec(idx)) / 60;
end

fig = figure;
bar(categorical(ageCats), meanFinishByAge);
ylabel('Average Finish Time (minutes)');
title('Average Finish Time by Age Group');
grid on;
saveFigure(fig, outputDir, '08_avg_finish_by_age_group.png');

% Average finish time by gender
genderCats = categories(data.gender);
meanFinishByGender = zeros(length(genderCats), 1);

for i = 1:length(genderCats)
    idx = data.gender == genderCats{i};
    meanFinishByGender(i) = mean(data.finish_net_sec(idx)) / 60;
end

fig = figure;
bar(categorical(genderCats), meanFinishByGender);
ylabel('Average Finish Time (minutes)');
title('Average Finish Time by Gender');
grid on;
saveFigure(fig, outputDir, '09_avg_finish_by_gender.png');

% Over/under threshold accuracy
fig = figure;
bar(betLines_min, betAccuracies);
xlabel('Benchmark Finish Time Line (minutes)');
ylabel('Over/Under Prediction Accuracy (%)');
title('Threshold-Based Over/Under Prediction Accuracy');
grid on;
saveFigure(fig, outputDir, '10_over_under_accuracy.png');

% Actual vs predicted under rates
fig = figure;
plot(betLines_min, actualUnderRates, '-o', 'LineWidth', 1.5);
hold on;
plot(betLines_min, predictedUnderRates, '-s', 'LineWidth', 1.5);
xlabel('Benchmark Finish Time Line (minutes)');
ylabel('Under Rate (%)');
title('Actual vs Predicted Under Rates');
legend('Actual Under Rate', 'Predicted Under Rate', 'Location', 'best');
grid on;
hold off;
saveFigure(fig, outputDir, '11_actual_vs_predicted_under_rates.png');

%% ============================================================
% 15. Final Presentation Talking Points
% ============================================================

fprintf('\n================ FINAL PRESENTATION SUMMARY ================\n');

fprintf('\nLatest achievements since Progress Update #2:\n');
fprintf(' - Completed a full end-to-end ML pipeline from data cleaning to saved outputs.\n');
fprintf(' - Added stronger model comparison using MAE, RMSE, R^2, MAPE, median absolute error, and within-time accuracy.\n');
fprintf(' - Added optional ensemble regression to compare against linear regression and neural network models.\n');
fprintf(' - Added final prediction export, saved model package, saved figures, and a quick demo prediction.\n');
fprintf(' - Converted regression output into an over/under threshold decision demo.\n');

fprintf('\nLessons learned:\n');
fprintf(' - Half-marathon split time is the strongest predictor of final marathon time.\n');
fprintf(' - Simple models can perform very strongly when the main feature has a clear physical relationship to the target.\n');
fprintf(' - Model evaluation is easier to explain when errors are converted from seconds into minutes.\n');
fprintf(' - A clean pipeline and reproducible outputs matter as much as the model itself.\n');

fprintf('\nTechnical obstacles and how we overcame them:\n');
fprintf(' - Categorical variables needed consistent encoding between training and testing sets.\n');
fprintf('   Solution: one-hot encoding was built using full dataset category definitions.\n');
fprintf(' - Raw race data could include missing or invalid time entries.\n');
fprintf('   Solution: missing rows, invalid times, and unrealistic finish/half ratios were removed.\n');
fprintf(' - Neural network inputs needed proper scaling.\n');
fprintf('   Solution: standardization parameters were learned only from training data and reused on testing data.\n');
fprintf(' - Regression results were difficult to explain directly in seconds.\n');
fprintf('   Solution: results are reported in minutes and converted into practical interpretation metrics.\n');

fprintf('\nIndividual contributions:\n');
fprintf(' - Sanad Sahawneh: ML pipeline development, model evaluation, visualization, and final system integration.\n');
fprintf(' - Brian Allen: dataset preparation support, performance interpretation, and presentation/demo preparation.\n');
fprintf(' - Mahib Rahman: model comparison support and results validation.\n');


fprintf('\nAll final outputs saved in folder: %s\n', outputDir);
fprintf('============================================================\n');

%% ============================================================
% Local Functions
% ============================================================

function metrics = calcMetrics(yTrue, yPred)
    err = yTrue - yPred;
    absErr = abs(err);

    metrics.MAE = mean(absErr);
    metrics.RMSE = sqrt(mean(err.^2));
    metrics.MAPE = mean(abs(err ./ yTrue)) * 100;
    metrics.R2 = 1 - sum(err.^2) / sum((yTrue - mean(yTrue)).^2);
    metrics.MAE_min = metrics.MAE / 60;
    metrics.RMSE_min = metrics.RMSE / 60;
    metrics.MedianAE_min = median(absErr) / 60;
    metrics.Within5min_pct = mean(absErr <= 5*60) * 100;
    metrics.Within10min_pct = mean(absErr <= 10*60) * 100;
    metrics.Within15min_pct = mean(absErr <= 15*60) * 100;
end

function [XtrainStd, XtestStd, featureNames, normParams] = buildFeatureMatrix(trainData, testData, allData)
    ageCats = categories(allData.age_group);
    genderCats = categories(allData.gender);

    ageTrain = categorical(trainData.age_group, ageCats);
    ageTest  = categorical(testData.age_group, ageCats);

    genderTrain = categorical(trainData.gender, genderCats);
    genderTest  = categorical(testData.gender, genderCats);

    Xtrain_age = dummyvar(ageTrain);
    Xtest_age  = dummyvar(ageTest);

    Xtrain_gender = dummyvar(genderTrain);
    Xtest_gender  = dummyvar(genderTest);

    Xtrain_num = trainData.half_time_sec;
    Xtest_num  = testData.half_time_sec;

    Xtrain = [Xtrain_age Xtrain_gender Xtrain_num];
    Xtest  = [Xtest_age Xtest_gender Xtest_num];

    featureNames = [strcat("age_", string(ageCats)); ...
                    strcat("gender_", string(genderCats)); ...
                    "half_time_sec"];

    mu = mean(Xtrain, 1);
    sigma = std(Xtrain, 0, 1);
    sigma(sigma == 0) = 1;

    XtrainStd = (Xtrain - mu) ./ sigma;
    XtestStd  = (Xtest  - mu) ./ sigma;

    normParams.mu = mu;
    normParams.sigma = sigma;
    normParams.ageCats = ageCats;
    normParams.genderCats = genderCats;
end

function predSec = predictSingleRunner(bestModelName, bestModelObject, runner, avgRatio, featureNames, normParams, allData)
    if strcmp(bestModelName, 'Baseline Ratio')
        predSec = runner.half_time_sec * avgRatio;
        return;
    end

    if strcmp(bestModelName, 'Linear Regression')
        demoTable = table( ...
            categorical(runner.age_group, categories(allData.age_group)), ...
            categorical(runner.gender, categories(allData.gender)), ...
            runner.half_time_sec, ...
            'VariableNames', {'age_group','gender','half_time_sec'});

        predSec = predict(bestModelObject, demoTable);
        return;
    end

    Xdemo = encodeSingleRunner(runner, normParams);
    predSec = predict(bestModelObject, Xdemo);
end

function XdemoStd = encodeSingleRunner(runner, normParams)
    ageCats = normParams.ageCats;
    genderCats = normParams.genderCats;

    ageOneHot = zeros(1, length(ageCats));
    genderOneHot = zeros(1, length(genderCats));

    ageIdx = find(strcmp(string(ageCats), string(runner.age_group)), 1);
    genderIdx = find(strcmp(string(genderCats), string(runner.gender)), 1);

    if isempty(ageIdx)
        error('Demo runner age group "%s" was not found in training categories.', runner.age_group);
    end

    if isempty(genderIdx)
        error('Demo runner gender "%s" was not found in training categories.', runner.gender);
    end

    ageOneHot(ageIdx) = 1;
    genderOneHot(genderIdx) = 1;

    Xdemo = [ageOneHot genderOneHot runner.half_time_sec];
    XdemoStd = (Xdemo - normParams.mu) ./ normParams.sigma;
end

function saveFigure(fig, outputDir, fileName)
    filePath = fullfile(outputDir, fileName);

    try
        exportgraphics(fig, filePath, 'Resolution', 300);
    catch
        saveas(fig, filePath);
    end
end

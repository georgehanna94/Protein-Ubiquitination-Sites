%Author: George Hanna
%Ensemble Method using trees

%% Train Ensemble With Unequal Classification Costs

%Training_data = csvread('G:\Google Drive\School\Fourth Year\BIOM5405 Project\Data\30_70split\70pcent_trainset_SMOTE_randomized.csv',34442,0,');
size(hepatitis)
VarNames = {'dieOrLive' 'age' 'sex' 'steroid' 'antivirals' 'fatigue' ...
    'malaise' 'anorexia' 'liverBig' 'liverFirm' 'spleen' ...
    'spiders' 'ascites' 'varices' 'bilirubin' 'alkPhosphate' 'sgot' ...
    'albumin' 'protime' 'histology'};
%%
% |hepatitis| is a 1-by-20 cell array of character vectors. The cells correspond to
% the response (|liveOrDie|) and 19 heterogeneous predictors.
%%
% Specify a numeric matrix containing the predictors and a cell vector
% containing |'Die'| and |'Live'|, which are response categories. The
% response contains two values: |1| indicates that a patient died, and |2|
% indicates that a patient lived. Specify a cell array of character vectors
% for the response using the response categories. The first variable in
% |hepatitis| contains the response.
X = cell2mat(hepatitis(2:end));
ClassNames = {'Die' 'Live'};
Y = ClassNames(hepatitis{:,1});
%%
% |X| is a numeric matrix containing the 19 predictors. |Y| is a cell array
% of character vectors containing the response.
%%
% Inspect the data for missing values.
figure;
barh(sum(isnan(X),1)/size(X,1));
h = gca;
h.YTick = 1:numel(VarNames) - 1;
h.YTickLabel = VarNames(2:end);
ylabel 'Predictor';
xlabel 'Fraction of missing values';
%%
% Most predictors have missing values, and one has nearly 45% of the
% missing values. Therefore, use decision trees with surrogate splits for
% better accuracy. Because the data set is small, training time with
% surrogate splits should be tolerable.
%%
% Create a classification tree template that uses surrogate splits.
rng(0,'twister') % for reproducibility
t = templateTree('surrogate','all');
%%
% Examine the data or the description of the data to see which predictors
% are categorical.
X(1:5,:)
%%
% It appears that predictors 2 through 13 are categorical, as well as
% predictor 19. You can confirm this inference using the data set
% description at <http://archive.ics.uci.edu/ml/datasets/Hepatitis UCI
% Machine Learning Data Repository>.
%%
% List the categorical variables.
catIdx = [2:13,19];
%%
% Create a cross-validated ensemble using 150 learners and the
% |GentleBoost| algorithm.
Ensemble = fitensemble(X,Y,'GentleBoost',150,t,...
  'PredictorNames',VarNames(2:end),'LearnRate',0.1,...
  'CategoricalPredictors',catIdx,'KFold',5);
figure;
plot(kfoldLoss(Ensemble,'Mode','cumulative','LossFun','exponential'));
xlabel('Number of trees');
ylabel('Cross-validated exponential loss');
%%
% Inspect the confusion matrix to see which patients the ensemble predicts
% correctly.
[yFit,sFit] = kfoldPredict(Ensemble);
confusionmat(Y,yFit,'Order',ClassNames)
%%
% Of the 123 patient who live, the ensemble predicts correctly that 112
% will live. But for the 32 patients who die of hepatitis, the ensemble
% only predicts correctly that about half will die of hepatitis.
%%
% There are two types of error in the predictions of the ensemble:
%
% * Predicting that the patient lives, but the patient dies
% * Predicting that the patient dies, but the patient lives
%
% Suppose you believe that the first error is five times worse than the
% second. Create a new classification cost matrix that reflects this
% belief.
cost.ClassNames = ClassNames;
cost.ClassificationCosts = [0 5; 1 0];
%%
% Create a new cross-validated ensemble using |cost| as the
% misclassification cost, and inspect the resulting confusion matrix.
EnsembleCost = fitensemble(X,Y,'GentleBoost',150,t,...
  'PredictorNames',VarNames(2:end),'LearnRate',0.1,...
  'CategoricalPredictors',catIdx,'KFold',5,...
  'Cost',cost);
[yFitCost,sFitCost] = kfoldPredict(EnsembleCost);
confusionmat(Y,yFitCost,'Order',ClassNames)
%%
% As expected, the new ensemble does a better job classifying thew patients
% who die. Somewhat surprisingly, the new ensemble also does a better job
% classifying the patients who live, though the result is not statistically
% significantly better. The results of the cross validation are random, so
% this result is simply a statistical fluctuation. The result seems to
% indicate that the classification of patients who live is not very sensitive to
% the cost.


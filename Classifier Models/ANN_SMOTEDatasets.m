%This will be the algorithm to work - GH
%Author: George Hanna and Jinny Lee
%BIOM 5405
%Course Project

clear all;
close all;

%Load SMOTE Training Set
trainset = csvread('\Data\40_60split\60p_trainset_SMOTE_K1.csv');
train_inputs = trainset(:,[1:435]);
train_targets = trainset(:,436);

testset = csvread('\Data\40_60split\40p_testset_full.csv');
test_inputs = testset(:,[1:435]);
test_targets = testset(:,436);

%Apply feature selection in WEKA and copy results here
%Features applied here are first 56 from greedy search
features = [234 214 256 233 209 217 172 257 411 239 230 211 238 246 216 244 259 229 138 126 245 243 218 114 415 124 179 212 60 299 396 161 171 58 372 119 421 173 392 213 387 165 362 436 155 361 198 391 412 433 115 65 164 166 194 420];
%features = [234 214 256 233 209 217 172 257 411 239];
traininputs = selectfeatures(features,train_inputs);
testinputs = selectfeatures(features,test_inputs);

x_test = testinputs;
t_test = test_targets;

%Neural Network Architecture with Hidden Layer size
hiddenLayerSize = 1;

%TrainBr implements bayesian regularization
net = patternnet(hiddenLayerSize, 'trainbr');
net.inputs{1}; %Line added to verify that patternet does NN normalization
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
net.trainParam.max_fail  =50;

%Training Phase
%Set number of NN to train
numNN = 10;
NN = cell(1, numNN);
perfs = zeros(1, numNN);

for i = 1:numNN
  fprintf('Training %d/%d\n', i, numNN);
  NN{i} = train(net, traininputs', train_targets');

  
end

%Testing Phase
perfs = zeros(1, numNN);
y2Total = 0 ;
for i = 1: numNN
    neti = NN{i};
    y2 = neti(x_test');
    perfs(i) = mse(neti,t_test',y2);
    y2Total = y2Total +y2;
end

%Calculate average outputs
y2AverageOutput = y2Total/numNN;
%Plots confusion matrix using actual target data and average output data.
figure, plotconfusion(t_test',y2AverageOutput)
perfAveragedOutputs = mse (NN{1},t_test,y2AverageOutput);



% % Create a Pattern Recognition Network (smaller number of layers to avoid
% % overfitting
% hiddenLayerSize = 5;
% net = patternnet(hiddenLayerSize);
% 
% 
% % Set up Division of Data for Training, Validation, Testing
% net.divideParam.trainRatio = 70/100;
% net.divideParam.valRatio = 15/100;
% net.divideParam.testRatio = 15/100;
% 
% 
% % Train the Network
% [net,tr] = train(net,inputs,targets);
% 
% % Test the Network
% outputs = net(inputs);
% errors = gsubtract(targets,outputs);
% performance = perform(net,targets,outputs);
% 
% %Test Network on test set
% tInd = tr.testInd;
% tstOutputs = net(inputs(:,tInd));
% tstPerform = perform(net,targets(:,tInd),tstOutputs);
% 
% % View the Network
% view(net)
% 
% % Plots
% % Uncomment these lines to enable various plots.
% % figure, plotperform(tr)
% % figure, plottrainstate(tr)
% figure, plotconfusion(targets,outputs)
% % figure, ploterrhist(errors)

%This will be the algorithm to work - GH
%Author: George Hanna and Jinny Lee
%BIOM 5405
%Course Project

clear all;
close all;

%Load Training Set
trainset = csvread('60p_trainset_full.csv');
train_inputs = trainset(:,[1:435]);
train_targets = trainset(:,436);

testset = csvread('40p_testset_full.csv');
test_inputs = testset(:,[1:435]);
test_targets = testset(:,436);

%Apply feature selection in WEKA and copy results here
%Features applied here are first 56 from greedy search
features = [234 214 256 233 209 217 172 257 411 239 230 211 238 246 216 244 259 229 138 126 245 243 218 114 415 124 179 212 60 299 396 161 171 58 372 119 421 173 392 213 387 165 362 436 155 361 198 391 412 433 115 65 164 166 194 420];
%features = [];
traininputs = selectfeatures(features,train_inputs);
testinputs = selectfeatures(features,test_inputs);

%Select Number of Boostrap Samples
nb_btrp_samples = 10;

positive = find(train_targets);
negative = find(train_targets==0);
NN = cell(1, nb_btrp_samples);

x_test = testinputs;
t_test = test_targets;

for k = 1:nb_btrp_samples
    iter_negative = randsample(negative,size(positive,1));
    iter_positive = randsample(positive,size(positive,1),true);
    trainset_targetinds = [iter_negative;positive];
    %Shuffle to remove bias
    trainset_targetinds = trainset_targetinds(randperm(length(trainset_targetinds)));
    desired_output  = train_targets(trainset_targetinds);
    trainset = traininputs(trainset_targetinds,:);
    
    %Split testset into 20% for testing, 20% for Validation 

    %Neural Network Architecture with Hidden Layer size
    hiddenLayerSize = 5;
    net = patternnet(hiddenLayerSize);
    net.trainFcn = 'trainbr';
%     net.trainParam.lr = 0.0005;
%     net.trainParam.mc = 0.3;
%     net.trainParam.max_fail = 50;
%     net.divideParam.trainRatio = 70/100;
%     net.divideParam.valRatio = 15/100;
%     net.divideParam.testRatio = 15/100;


    %Training Phase
    %Set number of NN to train
    perfs = zeros(1, nb_btrp_samples);
    
    fprintf('Training %d/%d\n', k, nb_btrp_samples);
    NN{k} = train(net, trainset', desired_output');

end

t = templateTree('surrogate','all');
tic
ensemble = fitensemble(traininputs,train_targets,'GentleBoost',150,t,...
   'LearnRate',0.1,'KFold',5);
toc

tic
results = predict(ensemble,testinputs);
toc
figure,plotconfusion(t_test',results')

%Testing Phase
perfs = zeros(1, nb_btrp_samples);
y2Total = 0 ;
for i = 1: nb_btrp_samples
    neti = NN{i};
    y2 = neti(x_test');
    perfs(i) = mse(neti,t_test',y2);
    y2Total = y2Total +y2;
end
perfs
y2AverageOutput = y2Total/nb_btrp_samples;
figure, plotconfusion(t_test',y2AverageOutput)
perfAveragedOutputs = mse (NN{1},t_test,y2AverageOutput);




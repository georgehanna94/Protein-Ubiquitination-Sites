%This will be the algorithm to work - GH
%Author: George Hanna and Jinny Lee
%BIOM 5405
%Course Project

clear all;
close all;

%Load Training Set and Oracle Set, then concatenate them
trainset = csvread('60p_trainset_full.csv');
oracledata = csvread('oracledata.csv');
trainset = vertcat(trainset,oracledata);
train_inputs = trainset(:,[1:435]);
train_targets = trainset(:,436);
train_inputs(train_inputs == -9999) = NaN;

%Load Holdout test set
testset = csvread('40p_testset_full.csv');
test_inputs = testset(:,[1:435]);
test_inputs(test_inputs == -9999) = NaN;
test_targets = testset(:,436);


%Handle -9999 Values
Med_devs = mad(train_inputs);
for i=1:size(train_inputs,2)
    
    %Replace NaNs with median in each column
    temp = train_inputs(:,i);
    k = find(isnan(temp));
    temp(k) = median(temp,'omitnan');
    train_inputs(:,i) = temp;
    
    %Replace NaNs with median in each column
    temp = test_inputs(:,i);
    temp(isnan(temp)) = median(temp, 'omitnan');
    test_inputs(:,i) = temp;
    
    %Replace train values more than 6 MADs away with median of column
    temp = abs(train_inputs(:,i)- median(train_inputs(:,i)))/Med_devs(i);
    Ind = find(temp>6);
    train_inputs(Ind,i) = median(train_inputs(:,i));
    
    %Replace test values more than 6 MADs away with median of column
    temp = abs(test_inputs(:,i)- median(test_inputs(:,i)))/Med_devs(i);
    Ind = find(temp>6);
    test_inputs(Ind,i) = median(test_inputs(:,i));

end

%Apply feature selection in WEKA and copy results here
%Features applied here are first 56 from greedy search
%features = [234 214 256 233 209 217 172 257 411 239 230 211 238 246 216 244 259 229 138 126 245 243 218 114 415 124 179 212 60 299 396 161 171 58 372 119 421 173 392 213 387 165 362];
features = [3,16,34,39,41,58,60,72,102,108,114,119,121,124,126,132,138,155,161,165,171,172,173,179,199,209,211,212,213,214,216,217,218,229,230,233,234,238,239,243,244,245,246,253,256,257,259,261,267,299,307,362,372,387,392,396,411,414,415,421,436];
traininputs = selectfeatures(features,train_inputs);
testinputs = selectfeatures(features,test_inputs);

%Select Number of Boostrap Samples
nb_btrp_samples = 10;

positive = find(train_targets);
negative = find(train_targets==0);
NN = cell(1, nb_btrp_samples);

t = templateTree('surrogate','all');


for k = 1:nb_btrp_samples
    iter_negative = randsample(negative,size(positive,1),true);
    iter_positive = randsample(positive,size(positive,1),true);
    trainset_targetinds = [iter_negative;positive];
    %Shuffle to remove bias
    trainset_targetinds = trainset_targetinds(randperm(length(trainset_targetinds)));
    desired_output  = train_targets(trainset_targetinds);
    trainset = traininputs(trainset_targetinds,:);
    
    %Create Ensemble Classifier
    %Ensemble{k} = fitensemble(trainset,desired_output,'RUSBoost',500,t,'LearnRate',0.1);
    
    %Neural Network Architecture with Hidden Layer size
    hiddenLayerSize = 3;
    net = patternnet(hiddenLayerSize);
    net.trainFcn = 'trainbr';

    %Training Phase
    %Set number of NN to train
    perfs = zeros(1, nb_btrp_samples);
    
    fprintf('Training %d/%d\n', k, nb_btrp_samples);
    NN{k} = train(net, trainset', desired_output');

end

%Temporary TEst
%Sample = fitensemble(trainset,desired_output,'LPBoost',500,t);
%temp = predict(Sample,x_test);
%figure, plotconfusion(t_test',temp')
%figure,plotconfusion(t_test',results')

%Testing Phase
perfs = zeros(1, nb_btrp_samples);
y2Total = 0 ;

%Number of Bootstrap test set samples used to reconstruct testing
%distribution
nb_test_set_samples = 200;
siteIds = 1:length(testinputs);
precision_95th = 1:nb_test_set_samples;
positive_tests = find(test_targets);
negative_tests = find(test_targets==0);

for j = 1: nb_test_set_samples
    
    iter_negative_tests = randsample(negative_tests,size(negative_tests,1),true);
    iter_positive_tests = randsample(positive_tests,size(positive_tests,1),true);
    testset_targetinds = [iter_negative_tests;iter_positive_tests];
    %Shuffle to remove bias
    %testset_targetinds = testset_targetinds(randperm(length(testset_targetinds)));
    t_test  = test_targets(testset_targetinds);
    x_test = testinputs(testset_targetinds,:);
    
    %Create a boostrap sample from the test set
    %x_test_inds = randsample(siteIds,size(siteIds,2),true);
    % x_test = testinputs(x_test_inds,:);
    %t_test = test_targets(x_test_inds,:);

    y2Total = 0;
    
    for i = 1: nb_btrp_samples
        neti = NN{i};
        %Apply NN
        y2 = neti(x_test');
        y2Total = y2Total +y2;
    end

    y2AverageOutput = y2Total / nb_btrp_samples;
    [Xpr,Ypr]= perfcurve(t_test, y2AverageOutput', 1, 'xCrit', 'reca', 'yCrit', 'prec');
    G= find(abs(Xpr-Ypr)<0.01);
    plot(Xpr,Ypr)
    precision_95th(j) = Ypr(G(end));
end

figure;
h1 = histogram(precision_95th);
h1.BinWidth = 0.005;

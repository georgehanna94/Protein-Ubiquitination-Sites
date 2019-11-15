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
%features = [234 214 256 233 209 217 172 257 411 239 230 211 238 246 216 244 259 229 138 126 245 243 218 114 415 124 179 212 60 299 396 161 171 58 372 119 421 173 392 213 387 165 362];
features = [3,16,34,39,41,58,60,72,102,108,114,119,121,124,126,132,138,155,161,165,171,172,173,179,199,209,211,212,213,214,216,217,218,229,230,233,234,238,239,243,244,245,246,253,256,257,259,261,267,299,307,362,372,387,392,396,411,414,415,421,436];
traininputs = selectfeatures(features,train_inputs);
testinputs = selectfeatures(features,test_inputs);

%Select Number of Boostrap Samples
nb_btrp_samples = 7;

positive = find(train_targets);
negative = find(train_targets==0);
NN = cell(1, nb_btrp_samples);

x_test = testinputs;
t_test = test_targets;
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
    hiddenLayerSize = 2;
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
for i = 1: nb_btrp_samples
    neti = NN{i};
    %Apply NN
    y2 = neti(x_test');
    y2 = round(y2);
    %Apply Ensemble Tree
    %results = predict(Ensemble{i},x_test);
    %perfs(i) = mse(neti,t_test',y2);
    y2Total = y2Total +y2; %+results';
    
    test_targets = test_targets';
    
    result{i} = sum(y2Total);
    if result{i} >= (nb_btrp_samples/5752)
        y2output{i} = result{i};
    else
        y2output = [];
    end
    
end

perfs
figure, plotconfusion(t_test',y2output)
perfAveragedOutputs = mse (NN{1},t_test,y2output);




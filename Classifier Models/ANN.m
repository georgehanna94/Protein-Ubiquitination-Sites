%This will be the algorithm to work - GH
%Author: George Hanna
%BIOM 5405
%Course Project


%Read in raw data (Done manually for now)
old_x = csvread('Input.csv');
t = csvread('Class.csv');

%Apply feature selection in WEKA and copy results here
features = [1 2 3];
x = selectfeatures(features,old_x);

%Split Data 60% for designing networks, 10% for testing, 30% for Validation
Q = size(x,1);
Q1 = floor(Q*0.90);
Q2 = Q - Q1;
%Add Randomization
ind = randperm(Q);
ind1=ind(1:Q1);
ind2 = ind(Q1+(1:Q2));
x1 = x(ind1, :);
t1 = t(ind1, :);
x2 = x(ind2, :);
t2 = t(ind2, :);

%SMOTE Training Data 




%Neural Network Architecture with Hidden Layer size
hiddenLayerSize = 1;
net = patternnet(hiddenLayerSize);

numNN = 10;
NN = cell(1, numNN);
perfs = zeros(1, numNN);
for i = 1:numNN
  fprintf('Training %d/%d\n', i, numNN);
  NN{i} = train(net, x1', t1');

end

perfs = zeros(1, numNN);
y2Total = 0 ;
for i = 1: numNN
    neti = nets{i};
    y2 = neti(x2);
    perfs(i) = mse(neti,t2,y2);
    y2Total = y2Total +y2;
end
perfs
y2AverageOutput = y2Total/numNN;
perfAveragedOutputs = mse (nets{1},t2,y2AverageOutput);



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

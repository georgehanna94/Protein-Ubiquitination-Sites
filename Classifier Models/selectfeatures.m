function [ newdata ] = selectfeatures( features, data )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if (isempty(features))
        newdata =data;
    
else 
    newdata = zeros([length(data) length(features)]);

    features = features - 1;

    for i=1:length(features)
        newdata(:,i)= data(:,features(i));
    end
end

end


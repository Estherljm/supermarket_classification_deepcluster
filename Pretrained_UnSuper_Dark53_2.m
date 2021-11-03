clear;clc;close all

%% Load Data
% 
dataFolder = 'C:\Users\esthe\OneDrive\Desktop\sp_train';

imds = imageDatastore(dataFolder, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

tbl = countEachLabel(imds);
minSetCount = min(tbl{:, 2});
imds = splitEachLabel(imds, minSetCount);

% Load Pretrained Network (DarkNet53)
net = darknet53();

inputSize = net.Layers(1).InputSize;
%analyzeNetwork(net);
augmentedSet = augmentedImageDatastore(inputSize,...
     imds,'ColorPreprocessing','gray2rgb');

%% Extract Image Features (DarkNet53)

% augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain);
% augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);

layer = 'avg1';
% featuresExt = activations(net, augmentedSet, layer, 'OutputAs', 'rows','MiniBatchSize', 32);
featuresExt = activations(net, augmentedSet, layer, 'OutputAs', 'rows','MiniBatchSize', 32);

whos featuresExt


%% Train K-means Unsupervised

LabelTest = imds.Labels;
allClass = unique(LabelTest);
num = numel(allClass);

[idx, clustering, sumD, D] = kmeans(featuresExt, num, 'Replicates', 100);

%%  load trained darknet / ready to deploy
load('unsuper_darknet53')

[val, ind] = min(D');
YTrain = ind';

XTrain = grp2idx(LabelTest);

nmi(XTrain, YTrain)
histogram(YTrain, 18)

[FileName, Path] = uigetfile ('*.bmp; *.png; *.jpg','Select the input image');
queryImage = imread(strcat(Path,FileName));

queryFeature = activations(net, queryImage, layer, 'OutputAs', 'rows');

querydistance = pdist2(queryFeature, featuresExt);

[q_val, q_ind] = mink(querydistance, 9);

thumbnailGallery = [];
for i = q_ind
    I = imread(imds.Files{i});
    thumbnailGallery = cat(4, thumbnailGallery, I);
end

montage(thumbnailGallery);

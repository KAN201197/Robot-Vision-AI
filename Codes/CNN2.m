%% clear workspace
clc
clear

%% load data
digitDatasetPath = fullfile('./', '/dataset/');
imds = imageDatastore(digitDatasetPath, ...
   'IncludeSubfolders',true,'LabelSource','foldernames');

% Seprate trainning and validation dataset
[imdsTrain,imdsValidation] = splitEachLabel(imds,0.75);

% Label Number
numClasses = numel(categories(imdsTrain.Labels));

% Count picture number of Trainning Dataset under each lable
TrainLabelCounter = countEachLabel(imdsTrain);
ValiLabelCounter = countEachLabel(imdsValidation);

a = numClasses
% Size of img
img=readimage(imds,1);
ImgSize = size(img)
%% preprocessing
% corrode

% rotation and scaling
%pixelRange = [-64 64]; %[-30,30]
scaleRange = [0.6 1.4];
rotateRange = [-90,90];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandYReflection',true, ...
    'RandRotation',rotateRange, ...
    'RandXScale',scaleRange, ...
    'RandYScale',scaleRange, ...
    'RandXTranslation',[-5 5],...
    'RandYTranslation',[-5 5] ...
    );

%     'RandXTranslation',pixelRange, ...
%     'RandYTranslation',pixelRange, ...

inputSize = [128 128 1];
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain ...
        ); % 'DataAugmentation',imageAugmenter

augimdsValidation = augmentedImageDatastore(inputSize(1:2),imdsValidation);

% minibatch = preview(augimdsTrain);
% imshow(imtile(minibatch.input));

%% Layers
numClasses = numel(categories(imdsTrain.Labels))

% layers = [
%  
%     imageInputLayer([128 128 1])
% 
%     convolution2dLayer(5,6,'Padding',2)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'stride',2)
%     convolution2dLayer(5, 16)
%     batchNormalizationLayer
%     reluLayer
%     maxPooling2dLayer(2,'stride',2)
%     convolution2dLayer(5, 128)
%     batchNormalizationLayer
%     reluLayer
%     
%     dropoutLayer(0.5)    
%     fullyConnectedLayer(numClasses)
%     softmaxLayer
%     classificationLayer
%     ];


layers = [
    imageInputLayer([128 128 1])
    convolution2dLayer(5, 8, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(5, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)    
    convolution2dLayer(5, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)   
    convolution2dLayer(5, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(5, 128, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(5, 256, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    convolution2dLayer(5, 512, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer(2, 'Stride', 2)
    fullyConnectedLayer(128)
    dropoutLayer(0.5)
    fullyConnectedLayer(32)
    dropoutLayer(0.5)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer
];

% visualise for debug
lgraph = layerGraph(layers);
analyzeNetwork(lgraph)


%% Training Network
% training options
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.001, ...    
    'MaxEpochs',40, ...
    'MiniBatchSize',64, ...
    'Shuffle','every-epoch', ...
    'ValidationData',augimdsValidation, ...
    'ValidationFrequency',5, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ValidationPatience',5 ...
    );                   
 
% training
net = trainNetwork(augimdsTrain,layers,options); 
save 'CNN.mat' net

YPred = classify(net,augimdsValidation);
YValidation = imdsValidation.Labels;

figure
confusionchart(YValidation,YPred)

fprintf('finish training')
%% store image

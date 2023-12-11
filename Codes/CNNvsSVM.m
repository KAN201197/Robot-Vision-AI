%% Wrokspace preparation
clear
clc

%% Data & model loading
CNN_Dir = fullfile('./','CNN.mat');
SVM_Dir = fullfile('./','SVM.mat');
Img_Dir = fullfile('./','result/');
data = fullfile('./','dataset/');
CNNmodel = load(CNN_Dir);
SVMmodel = load(SVM_Dir);
imds = imageDatastore(Img_Dir, ...
   'IncludeSubfolders',true,'LabelSource','foldernames');
dataset = imageDatastore(data,'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
% change the image into binary !!!!!, make sure [128,128,1]
dataimg = readimage(dataset, 2); 
dataImgSize = size(dataimg);

segimg = readimage(imds,2);
segImgSize = size(segimg);

fprintf('dataImageSize:%dx%d\n',dataImgSize(1),dataImgSize(2))
fprintf('segImageSize:%dx%d\n',segImgSize(1),segImgSize(2))

% Pre-processing
Model_inputSize = [128 128 1];
augimdsTest = augmentedImageDatastore(Model_inputSize(1:2),imds);

augimdsVali = augmentedImageDatastore(Model_inputSize(1:2),dataset);

%% Test
% Img = imread();
% Img = imresize(Img, [28 28]);   % Resize to match CNN input size
% % Img = rgb2gray(Img);            % Convert to grayscale
% Img = im2double(Img);           % Scale pixel values to [0, 1]
% 
% [label, score] = classify(CNNmodel.net,Img);
% figure;
% imshow(Img);title(label)

%% CNN
CNN_Result = classify(CNNmodel.net,imds);
CNN_Label = imds.Labels
CNN_ResultLable = CNN_Result

[YPred,probs] = classify(CNNmodel.net,imds);
accuracy = mean(YPred == imds.Labels)


%% SVM

% HOG
% single test picture
segImg = readimage(imds, 1);

% Extract HOG features and HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(segImg,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(segImg,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(segImg,'CellSize',[8 8]);

segCellSize = [4 4];
SegHogFeatureSize = length(hog_4x4);

[segtestFeatures, segtestLabels] = helperExtractHOGFeaturesFromImageSet(imds, SegHogFeatureSize, segCellSize);

% % PCA
% [~, scores, ~, ~, explained] = pca(segtestFeatures);
% numFeaturesToKeep = find(cumsum(explained)/sum(explained) >= 0.95, 1);
% segtestFeatures = scores(:, 1:numFeaturesToKeep);

% Make class predictions using the test features.
predictedLabels = predict(SVMmodel.classifier, segtestFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(segtestLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat) 
accuracy = mean(predictedLabels == imds.Labels)


%% Support function

function helperDisplayConfusionMatrix(confMat)
% Display the confusion matrix in a formatted table.

% Convert confusion matrix into percentage form
confMat = bsxfun(@rdivide,confMat,sum(confMat,2));

digits = ['0','4','7','8','A','D','H'];
colHeadings = arrayfun(@(x)sprintf('%c',x),digits,'UniformOutput',false);
format = repmat('%-9s',1,11);
header = sprintf(format,'ConfusionMatrix  |',colHeadings{:});
fprintf('\n%s\n%s\n',header,repmat('-',size(header)));
for idx = 1:numel(digits)
    fprintf('%-9s',   [digits(idx) '                |']);
    fprintf('%-9.2f', confMat(idx,:));
    fprintf('\n')
end
end

function [features, setLabels] = helperExtractHOGFeaturesFromImageSet(imds, hogFeatureSize, cellSize)
% Extract HOG features from an imageDatastore.

setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages,hogFeatureSize,'single');

% Process each image and extract features
for j = 1:numImages
    img = readimage(imds,j);
    %img = im2gray(img); % for no-rgb image.
    
    % Apply pre-processing steps
    img = uint8(img); % for logial image, while for normal image, just comment it.
    img = imbinarize(img);
    
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
end

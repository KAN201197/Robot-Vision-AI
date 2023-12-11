%% Workspace prepare
clc
clear

%% Load training and test data using |imageDatastore|.
charactersDir = fullfile('./', '/newdataset/');
segDir = fullfile('./','/NewSegmentResult/');
imds = imageDatastore(charactersDir, ...
   'IncludeSubfolders',true,'LabelSource','foldernames');% use filename as lable
SegTestImds = imageDatastore(segDir, ...
   'IncludeSubfolders',true,'LabelSource','foldernames');

% Seprate trainning and validation dataset
[trainingSet,testSet] = splitEachLabel(imds,0.75);

% Label Number
numClasses = numel(categories(trainingSet.Labels))

% Count picture number of Trainning Dataset under each lable
TrainLabelCounter = countEachLabel(trainingSet)
ValiLabelCounter = countEachLabel(testSet);


%% Showing some traning and testing images

% figure;
% 
% subplot(2,3,1);
% imshow(trainingSet.Files{102});
% 
% subplot(2,3,2);
% imshow(trainingSet.Files{6});
% 
% subplot(2,3,3);
% imshow(trainingSet.Files{18});
% 
% subplot(2,3,4);
% imshow(testSet.Files{13});
% 
% subplot(2,3,5);
% imshow(testSet.Files{37});
% 
% subplot(2,3,6);
% imshow(testSet.Files{97});

%% Showing and select of HOG features of specific picture

img = readimage(trainingSet, 40);

% Extract HOG features and HOG visualization
[hog_2x2, vis2x2] = extractHOGFeatures(img,'CellSize',[2 2]);
[hog_4x4, vis4x4] = extractHOGFeatures(img,'CellSize',[4 4]);
[hog_8x8, vis8x8] = extractHOGFeatures(img,'CellSize',[8 8]);

% Show the original image
figure; 
subplot(2,3,1:3); imshow(img);

% Visualize the HOG features
subplot(2,3,4);  
plot(vis2x2); 
title({'CellSize = [2 2]'; ['Length = ' num2str(length(hog_2x2))]});

subplot(2,3,5);
plot(vis4x4); 
title({'CellSize = [4 4]'; ['Length = ' num2str(length(hog_4x4))]});

subplot(2,3,6);
plot(vis8x8); 
title({'CellSize = [8 8]'; ['Length = ' num2str(length(hog_8x8))]});

%% Cell size set
cellSize = [4 4];
hogFeatureSize = length(hog_4x4);

%% Train a Digit Classifier

% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.

numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages,hogFeatureSize,'single');

for i = 1:numImages
    img = readimage(trainingSet,i);
    
    % img = im2gray(img);
    
    % Apply pre-processing steps
    % img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img,'CellSize',cellSize);  
end

% Get labels for each image.
trainingLabels = trainingSet.Labels;

% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
classifier = fitcecoc(trainingFeatures, trainingLabels)
save 'SVM.mat' classifier


%% Evaluate the Digit Classifier

% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);

% Make class predictions using the test features.
predictedLabels = predict(classifier, testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

helperDisplayConfusionMatrix(confMat)


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

function [features, setLabels] = helperExtractHOGFeaturesFromImageSet( ...
    imds, hogFeatureSize, cellSize)
% Extract HOG features from an imageDatastore.

setLabels = imds.Labels;
numImages = numel(imds.Files);
features  = zeros(numImages,hogFeatureSize,'single');

% Process each image and extract features
for j = 1:numImages
    img = readimage(imds,j);
%    img = im2gray(img);
    
    % Apply pre-processing steps
%    img = imbinarize(img);
    
    features(j, :) = extractHOGFeatures(img,'CellSize',cellSize);
end
end


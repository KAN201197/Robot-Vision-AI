# ME5411 Robot Vision & AI Project

This project has an objective to apply image processing technique which can improve image quality from original image showing some character on a microchip by reducing the noise and smoothing the image. It also use segmentation technique to extract the main feature on these objects such as converting to binary image, creating the outline of characters using edge detection method, and labelling each object which will be used later by machine learning algorithm for further analysis. The CNN (Convolutional Neural Network) and SVM (Support Vector Machine) used to create a classification system to analyze and classify each character on the original image and will be compared both of these apporaches in terms of effectiveness in classifying each character on the image. 

## Executable files

Below are listed the executable files and folders:

1. Codes
   Inside of this folder, it has a following codes:
   
   a. Interface.m
   
      A Matlab code showing the user interface for applying image processing and segmentation technique as well as applying training result from CNN and SVM to classify the character on original image.

    b. CNN2.m
   
      A Matlab code showing the train code for CNN algorithm
   
   c. SVM.m
   
      A Matlab code showing the train code for SVM algorithm
   
   d. CNNvsSVM.m
   
      A Matlab code showing the comparison of prediction result of CNN and SVM algorithm
   
   e. CNN.mat
   
      A Matlab data showing the trained network of CNN
   
   f. SVM.mat
   
      A Matlab data showing the Trained SVM model object

3. Data Set
   
   A folder showing the data set of each character which will be used by machine learning algorithm to trained the network model

5. Segmentation Result
   
   A folder showing the segmentation result after applying segmentation technique

7. charact2
   
   The original image showing the label of characters on a microchip

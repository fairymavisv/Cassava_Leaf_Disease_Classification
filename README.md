# Kaggle Competition: Cassava Leaf Disease Classification

## DateSet

21,367 labeled images collected during regular surveys in Uganda provided by Kaggle.

## Achievement

1. Balanced sampling：

  In the case where there are significantly more label 3 than other labels, balanced sampling is performed to improve the recognition accuracy 
  of a few classes and to prevent model overfitting.

2. Image pre-processing:
   
  - Geometric transformation: RandomVerticalFlip implements a random vertical flip of a given image with a given probability.
  - Image Enhancement: GaussianBlur implements blurring the image using a randomly selected gaussian blur.
  - Affine transform: Affine transform is a 2-dimensional linear transform consisting of 5 basic operations, namely, rotation, translation, 
    scaling, misalignment, and flip.

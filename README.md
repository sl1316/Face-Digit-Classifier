# Face-Digit-Classifier
#contains 3 classifier: 
naiveBayes, perceptron, and svm classifier. SVM performs 100% accuracy in face recogintion and 93% in digit #recognition with cross-validation.
#contains 2 features: 
1. Hog feature 
2. Combination of 6 kinds of features. The first is “upper” which is 1 when upper half has more points. The second is “left” which is 1 when left half has more points. The third is “Horizontal Symmetry” which is “1” when left and right sides are symmetry. The fourth is “regions” which is 1 when there are exactly i connected regions in image. The fifth is “empty” which is 1 when ith line is empty. The last one is “hole” which is 1 when ith line has a hole of empty pixels. 


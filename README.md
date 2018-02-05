#DoubleML
The DoubleML package is an implementation of the estimation procedure developed in "Double Machine 
Learning for Treatment and Causal Parameters" by Victor Chernozhukov, Denis Chetverikov, Mert Demirer,
Esther Duflo, Christian Hansen, and Whitney Newey. 

The Double ML package has options to implement the double mahcine learning estimation procedure with
the following machine learning techniques: regression trees, boosted trees, random forest, lasso,
ridge, or logistic regressionswith an L-1 or L-2 penalty parameter.

The package contains 3 classes: ML2_Estimator, LassoLogitCV, and RidgeLogitCV.

The [ML2Estimator](DoubleML-Python/ML2Estimator_documentation.md? "ML2Estimator Documentation") class implements both the partial linear estimator and the interactive estimator outlined in Chernozhukov et. al.

The [LassoLogitCV](DoubleML-Python/LassoLogitCV_documentation.md? "LassoLogitCV Documentation") class is an implementation of a logistic regression with an L-1 penalty parameter chosen by leave-one-out cross validation. 

The [RidgeLogitCV](DoubleML-Python/RidgeLogitCV_documentation.md? "RidgeLogitCV Documentation") class is an implementation of a logistic regression with an L-2 penatly parameter chosen by leave-one-out cross validation

Installation instructions:
  TO DO

#DoubleML
The DoubleML package is an implementation of the estimation procedure developed in "Double Machine 
Learning for Treatment and Causal Parameters" by Victor Chernozhukov, Denis Chetverikov, Mert Demirer,
Esther Duflo, Christian Hansen, and Whitney Newey. 

The Double ML package has options to implement the double mahcine learning estimation procedure with
the following machine learning techniques: regression trees, boosted trees, random forest, lasso,
ridge, or logistic regressionswith an L-1 or L-2 penalty parameter.

The package contains 3 classes: ML2_Estimator, LassoLogitCV, and RidgeLogitCV.

The [ML2Estimator](https://github.com/kquist/DoubleML-Python/blob/master/ML2Estimator_documentation.md "ML2Estimator Documentation") class implements both the partial linear estimator and the interactive estimator outlined in Chernozhukov et. al.

The [LassoLogitCV](https://github.com/kquist/DoubleML-Python/blob/master/LassoLogitCV_documentation.md "LassoLogitCV Documentation") class is an implementation of a logistic regression with an L-1 penalty parameter chosen by leave-one-out cross validation. 

The [RidgeLogitCV](https://github.com/kquist/DoubleML-Python/blob/master/RidgeLogitCV_documentation.md "RidgeLogitCV Documentation") class is an implementation of a logistic regression with an L-2 penatly parameter chosen by leave-one-out cross validation

The file [example.py](https://github.com/kquist/DoubleML-Python/blob/master/example.py "example") is an example of how to implement the double machine leanring method. This example uses [sipp1991.dta](https://github.com/kquist/DoubleML-Python/blob/master/sipp1991.dta "data"), which is the data used in Chernozhukov and Hansen
(2004), to estimate the effect of 401(k) eligibility on net financial assets. See "Double Machine 
Learning for Treatment and Causal Parameters" for more information.

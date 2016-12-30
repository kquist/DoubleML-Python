# ML2
The DoubleML package is intended to be used to implement the estimation procedure developed in "Double Machine 
Learning for Treatment and Causal Parameters" by Victor Chernozhukov, Denis Chetverikov, Mert Demirer,
Esther Duflo, Christian Hansen, and Whitney Newey. The package contains implementations of 

	The package contains 3 classes: ML2_Estimator, LassoLogitCV, and RidgeLogitCV.

	The ML2_Estimator class implements both the partial linear estimator and the interactive estimator outlined in Chernozhukov et. al.
	This class can use a variety of machine learning  echniques (Regression Trees, Ada Boosted Trees, Random Forest, Lasso,
	Ridge, and Logistic regression with either an L-1 penalty parameter or an L-2 penalty parameter).

	The LassoLogitCV class is an implementation of a logistic regression with an L-1 penalty parameter chosen by leave-one-out cross validatoion

	The RidgeLogitCV class is an implementation of a logistic regression with an L-2 penatly parameter chosen by leave-one-out cross validation

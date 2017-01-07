#Class: RidgeLogitCV
The RidgeLogitCV class is an implementation of an estimation of a logistic regression with an L-2 penalty parameter chosen using k-fold leave-one-out cross validation
#Model Explanation
A logistic regression model is a linear model for classification also referred to as a "logit" model. In this model, there are two possible outcomes, 1 and 0. The probability that outcome "1" occurs given X and beta is ![Alt text](Logit_Generating.png?raw=true "Logit Model")
The class RidgeLogitCV uses a maximum likelihood estimator with an L-2 penalty parameter to estimate the values of beta. Notice that the log-likelihood of an outcome Y given X, and beta are as follows:
![Alt text](Logit_LL_Deriv.png?raw=true "Title")
#Parameters:
![Alt text](RidgeLogitMinimize.png?raw=true "Title")

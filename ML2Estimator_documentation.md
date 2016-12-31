#Class: ML2_Estimator
#Parameters:

- method: str, optional (default="Tree"). This determines which machine learning estimation technique will be employed for estimates with a non-binary outcome variable. Options include : "Tree" (for regression tree), "Boosted Tree" (for ada boosted tree), "Random Forest", "Lasso", and "Ridge"

- method_binary: str or none, optional (default=None). This determines which machine learning estimation technique will be employed for estimates with a non-binary outcome variable. Options include : "Tree" (for regression tree), "Boosted Tree" (for ada boosted tree), "Random Forest", "Lasso", "Ridge", "Lasso Logit" (for a logistic regression with an L-1 penalty parameter), and "Ridge Logit" (for a logistic regression with an L-2 penalty parameter). If method is set to "Tree", "Boosted Tree", or "Random Forest", then method_binary will default to the same value as method. However, if method is set to "Logit" or "Ridge", then method_binary will default to "Lasso Logit" or "Ridge" respectively.

- method_options: dict or none, optional (default=None). These are the options specific to the machine learning method defined in the 'method' parameter

- method_optiions_binary: These are the options specific to the machine learning method defined in the 'method' parameter

#Attributes:

- method_class: The initialized instance of the class of mahcine learning method that self.method corresponds to. For example, if 'self.method'="Random Forest", then 'self.method_class' will be an instance of a Random Forest estimator with the settings defined by 'method_options'.

- method_class_binary: The initialized instance of the class of mahcine learning method that self.method_binary corresponds to. For example, if 'self.method_binary'="Ridge Logit", then 'self.method_class_binary' will be an instance of a Random Forest estimator with the settings defined by 'method_options_binary'.

- pl_beta: The estimated effect of d on the outcome variable using the partial linear estimation strategy

- pl_se: The standard error of the estimate of the effect of d on the outcome variable using the partial linear estimation strategy:

- interactive_beta: The estimated effect of d on the outcome variable using the partial linear estimation strategy

- interactive_se: The standard error of the estimate of the effect of d on the outcome variable using the partial linear estimation strategy:

#Methods:

- define_lasso(self,binary_outcome): Initializes and returns an instance of the sklearn.linear_model.LassoCV class. If binary_outcome is True, then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

- define_lasso_logit(self,binary_outcome): Initializes and returns  an instance of the LassoLogitCV class.If binary_outcome is True, then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

- define_random_forest(self,binary_outcome): Initializes and returns an instance of the sklearn.ensemble.RandomForestRegressor class.If binary_outcome is True, then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

- define_regression_tree(self,binary_outcome): Initializes and returns  an instance of sklearn.model_selection.GridSearchCV() class where tree.DecisionTreeRegressor is the model being searched over.If binary_outcome is True, then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

- define_regression_tree_boosted(self,binary_outcome): Initializes and returns  an instance of the sklearn.ensemble.AdaBoostRegressor class.If binary_outcome is True,then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

- define_ridge(self,binary_outcome): Initializes and returns  an instance of the sklearn.linear_model.RidgeCV class. If binary_outcome is True, then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

- define_ridge_logit(self,binary_outcome): Initializes and returns an instance of the RidgeLogitCV class. If binary_outcome is True, then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

- define_model(self,binary_outcome): If binary_outcome is  false, this method will initialize and return a member of class type self.method with self.method_options If binary_outcome is False, this method will initialize and return a member of class type self.method_binary with self.method_options_binary

- fit(self,X,Y,binary_outcome): This method fits a machine learning method using regressors X and the outcome variable Y. If binary_outcome is True, then self.method_class_binary will be the ml method used. If binary_outcome is False, then self.method_class will be used. X should be an mxn numpy array whwere m is the number of observations and n are the regressors. Y should be 1-d numpy array of length m
		
- find_residuals(self, y_use,y_out,x_use,x_out,binary_outcome): This method uses x_use and y_use to fit a model, and then uses that calculates yhat=E[y|x_out]. This method returns yhat and its residuals (y_out-yhat)

- pl_estimate(self,X,y,d,test_size, normalize,second_order_terms verbose, standard_errors):: This method is the implementation of the double machine learning partial linear estimation explained in Chernozhukov et. al. This method returns the class with the beta estimate stored in self.PL_beta and the standard error stored in self.PL_se

- interactive_estimate(self,X,y,d,test_size,normalize, second_order_terms, drop_zero_divide, modify_zero_divide,verbose): This method is the implementation of the double machine learning interactive estimation explained in Chernozhukov et. al. This method returns the class with the beta estimate stored in self.Interactive_beta and the standard errorstored in self.Interactive_se

#Valid dictionary options for method_options and method_options_binary

This section includes the options for the [Tree](#tree), [Boosted Tree](#boosted),[Random Forest](#random), [Ridge](#ridge), [RidgeLogit](#ridgelogit), [Lasso](#lasso), and [LassoLogit](#lassologit)

Tree
=======
Boosted Tree
=======
Random Forest
=======
RidgeLogit
=======
Ridge 
=======
Lasso
=======
Our Lasso is an implementation of sklearn's LassoCV class found [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV). The L-1 penalty parameter is chosen using k-fold cross validation. The valid method-options are as follows:
LassoLogit
=======



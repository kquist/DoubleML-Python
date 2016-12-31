#Class: ML2_Estimator
#Parameters:

- method: str, optional (default="Tree")

- method_binary: str or none, optional (default=None)

- method_options: dict or none, optional (default=None)

- method_optiions_binary:

#Attributes:

- method_class: The initialized instance of the class of mahcine learning method that self.method corresponds to. For example, if 'self.method'="Random Forest", then 'self.method_class' will be an instance of a Random Forest estimator with the settings defined by 'method_options'.

- method_class_binary: The initialized instance of the class of mahcine learning method that self.method_binary corresponds to. For example, if 'self.method_binary'="Ridge Logit", then 'self.method_class_binary' will be an instance of a Random Forest estimator with the settings defined by 'method_options_binary'.

- pl_beta: The effect of d on the outcome variable using the partial linear estimation strategy

- pl_se: The standard error of the effect of d on the outcome variable using the partial linear estimation strategy:

- interactive_beta: The effect of d on the outcome variable using the partial linear estimation strategy

- interactive_se: The standard error of the effect of d on the outcome variable using the partial linear estimation strategy:

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

- pl_Estimate(self,X,y,d,test_size, normalize,second_order_terms verbose, standard_errors):: This method is the implementation of the double machine learning partial linear estimation explained in Chernozhukov et. al. This method returns the class with the beta estimate stored in self.PL_beta and the standard error stored in self.PL_se

- interactive_Estimate(self,X,y,d,test_size,normalize, second_order_terms, drop_zero_divide, modify_zero_divide,verbose): This method is the implementation of the double machine learning interactive estimation explained in Chernozhukov et. al. This method returns the class with the beta estimate stored in self.Interactive_beta and the standard errorstored in self.Interactive_se

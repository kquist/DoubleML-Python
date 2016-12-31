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

This section includes the options for the [Tree](#tree), [Boosted Tree](#boosted-tree),[Random Forest](#random-forest), [Ridge](#ridge regression), [RidgeLogit](#ridge-logit), [Lasso](#lasso), and [Lasso Logit](#lasso-logit)

Tree
=======
Boosted Tree
=======
Random Forest
=======
Ridge Logit
=======
Ridge 
=======
Lasso
=======
The Lasso option is an implementation of sklearn's LassoCV class found [here](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV). The L-1 penalty parameter is chosen using k-fold cross validation. The valid method_option dictionary entries are as follows:

- eps : float, optional. Default=1E-3. Length of the path. i.e. "eps=1e-3" means that "alpha_min/alpha_max=1e-3".
		    
- n_alphas : int, optional. Default=100. Number of alphas along the regularization path

- alphas : numpy array, optional, Default=None. List of alphas where to compute the models. If ``None`` alphas are set automatically
		    
- precompute : True | False | 'auto' | array-like. Default='auto'. Whether to use a precomputed Gram matrix to speed up calculations. If set to ``'auto'`` let us decide. The Gram matrix can also be passed as argument.
		    
- max_iter : int, optional. Default=5000. The maximum number of iterations

- tol : float, optional. Default=1E-4. The tolerance for the optimization: if the updates are smaller than "tol", the optimization code checks the dual gap for optimality and continues until it is smaller than "tol".

- cv : int, cross-validation generator or an iterable, optional. Default=10. 
	Determines the cross-validation splitting strategy.
	Possible inputs for cv are:
		        - integer, to specify the number of folds.
		        - An object to be used as a cross-validation generator.
		        - An iterable yielding train/test splits.
		        For integer/None inputs, :class:`KFold` is used.
		        Refer :ref:`User Guide <cross_validation>` for the various
		        cross-validation strategies that can be used here.
		    verbose : bool or integer. Default=False
		        Amount of verbosity.
		    n_jobs : integer, optional. Default=1
		        Number of CPUs to use during the cross validation. If ``-1``, use
		        all the CPUs.
		    positive : bool, optional. Default=False
		        If positive, restrict regression coefficients to be positive
		    selection : str, default 'cyclic'
		        If set to 'random', a random coefficient is updated every iteration
		        rather than looping over features sequentially by default. This
		        (setting to 'random') often leads to significantly faster convergence
		        especially when tol is higher than 1e-4.
		    random_state : int, RandomState instance, or None (default)
		        The seed of the pseudo random number generator that selects
		        a random feature to update. Useful only when selection is set to
		        'random'.
		    fit_intercept : boolean, default True
		        whether to calculate the intercept for this model. If set
		        to false, no intercept will be used in calculations
		        (e.g. data is expected to be already centered).
		    normalize : boolean, optional, default True
		        If ``True``, the regressors X will be normalized before regression.
		        This parameter is ignored when ``fit_intercept`` is set to ``False``.
		        When the regressors are normalized, note that this makes the
		        hyperparameters learnt more robust and almost independent of the number
		        of samples. The same property is not valid for standardized data.
		        However, if you wish to standardize, please use
		        :class:`preprocessing.StandardScaler` before calling ``fit`` on an estimator
		        with ``normalize=False``.
		    copy_X : boolean, optional, default True
		        If ``True``, X will be copied; else, it may be overwritten.
Lasso Logit
=======



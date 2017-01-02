#Class: ML2_Estimator
The DoubleML package is intended to be used to implement the estimation procedure developed in ["Double Machine 
Learning for Treatment and Causal Parameters"](https://arxiv.org/abs/1608.00060) by Victor Chernozhukov, Denis Chetverikov, Mert Demirer,
Esther Duflo, Christian Hansen, and Whitney Newey
#Parameters:

- method: str, optional (default="Tree"). This determines which machine learning estimation technique will be employed for estimates with a non-binary outcome variable. Options include : "Tree" (for regression tree), "Boosted Tree" (for ada boosted tree), "Random Forest", "Lasso", and "Ridge"

- method_binary: str or none, optional (default=None). This determines which machine learning estimation technique will be employed for estimates with a non-binary outcome variable. Options include : "Tree" (for regression tree), "Boosted Tree" (for ada boosted tree), "Random Forest", "Lasso", "Ridge", "Lasso Logit" (for a logistic regression with an L-1 penalty parameter), and "Ridge Logit" (for a logistic regression with an L-2 penalty parameter). If method is set to "Tree", "Boosted Tree", or "Random Forest", then method_binary will default to the same value as method. However, if method is set to "Logit" or "Ridge", then method_binary will default to "Lasso Logit" or "Ridge" respectively.

- method_options: dict or none, optional (default=None). These are the options specific to the machine learning method defined in the 'method' parameter. For more information about available method_options see [here](#available-machine-learning-methods).

- method_optiions_binary: These are the options specific to the machine learning method defined in the 'method_binary' parameter see [here](#available-machine-learning-methods).

#Attributes:

- method_class: The initialized instance of the class of mahcine learning method that self.method corresponds to. For example, if 'self.method'="Random Forest", then 'self.method_class' will be an instance of a Random Forest estimator with the settings defined by 'method_options'.

- method_class_binary: The initialized instance of the class of mahcine learning method that self.method_binary corresponds to. For example, if 'self.method_binary'="Ridge Logit", then 'self.method_class_binary' will be an instance of a Random Forest estimator with the settings defined by 'method_options_binary'.

- pl_beta: The estimated effect of d on the outcome variable using the partial linear estimation strategy

- pl_se: The standard error of the estimate of the effect of d on the outcome variable using the partial linear estimation strategy:

- interactive_beta: The estimated effect of d on the outcome variable using the partial linear estimation strategy

- interactive_se: The standard error of the estimate of the effect of d on the outcome variable using the partial linear estimation strategy:

#Methods:



- fit(self,X,Y,binary_outcome): This method fits a machine learning method using regressors X and the outcome variable Y. The parameters are as follows: 
		
		X:mxn numpy array whwere m is the number of observations and n are the regressors.
		
		Y:1-d numpy array of length m representing the outcome variable.
		
		Binary_outcome: boolean. If false, the machine learning method specified in self.method is used, if 
			set to True, then the machine learning method specified in self.method_binary is used.

- pl_estimate(self,X,y,d,test_size, normalize,second_order_terms verbose, standard_errors):: This method is the implementation of the double machine learning partial linear estimation explained in Chernozhukov et. al. This method estimates the beta coefficient of the binary regressor d on the outcome variable y when other regressors X are correlated with both X and y. This method returns the class with the beta estimate stored in self.pL_beta and the estimate's standard error stored in self.pL_se. The parameters are as follows

		X: mxn numpy array where m is the number of observations and n is the number of regressors.
		
		y: numpy row vector of length m where y[i] corresponds to x[:,i]
		
		d: numpyrow vector of length m where d[i] corresponds to x[:,i]
		
		test_size : float, int (default=.5)
			If float, should be between 0.0 and 1.0 and represent the
        		proportion of the dataset to include in the test split. If
        		int, represents the absolute number of test samples. If None,
        		the value is automatically set to the complement of the train size.
        		If train size is also None, test size is set to 0.25.
        	
		normalize: boolean, optional (default=True).
        		If set to true, each regressor is normalized to have a standard deviation of 1 across the sample.
        		This is recommended for both lasso and ridge methods
        	
		second_order_terms: boolean, optional (default=False)
        		If set to true, then the machine learning method uses both all of the regressors included in X,
        		and their second order terms (each regressor squared and interactive effects).
		
		verbose: boolean, optional (default=True).
			If set to true, then the beta and standard error results will be printed 
		
		standard_errors: string, optional (default="White")
			Options:
				-"Normal": results in normal standard errors
				-"White": results in heteroskedasticity robust standard errors
					as in White 1980
				-"Mackinnon": results in alternative heteroskedasticity robust standard errors
					as in Mackinnnon and White 1985

- interactive_estimate(self,X,y,d,test_size,normalize, second_order_terms, drop_zero_divide, modify_zero_divide,verbose): This method is the implementation of the double machine learning interactive estimation explained in Chernozhukov et. al. This method returns the class with the beta estimate stored in self.Interactive_beta and the standard errorstored in self.interactive_se


		X: mxn numpy array where m is the number of observations and n is the number of regressors.
		
		y: numpy row vector of length m where y[i] corresponds to x[:,i]
		
		d: numpyrow vector of length m where d[i] corresponds to x[:,i]
		
		test_size : float, int (default=.5)
        		If float, should be between 0.0 and 1.0 and represent the
        		proportion of the dataset to include in the test split. If
        		int, represents the absolute number of test samples. If None,
        		the value is automatically set to the complement of the train size.
        		If train size is also None, test size is set to 0.25.
        	
		normalize: boolean, optional (default=True).
        		If set to true, each regressor is normalized to have a standard deviation of 1 across the sample.
        		This is strongly recommended for both lasso and ridge methods
        
		second_order_terms: boolean, optional (default=False)
        		If set to true, then the machine learning method uses both all of the regressors included in X,
        		and their second order terms (each regressor squared and interactive effects).
        	
		drop_zero_divide: boolean, optional (default=False). 
        		If the actual value of d_out[i] is 1 but the predicted value of dhat[i] is 0 (or visa versa),
        		then the interactive estimate will necessarily have a divide by zero error.If drop_zero_divide
        		is True, then all cases in which a divide by zero error would occur will be thrown out of the sample
		
		verbose: boolean, optional (default=True).
			If set to true, then the beta and standard error results will be printed 
		
		modify_zero_divide: float, optional (default=1E-3). modify_zero_divide is only used if drop_zero_divide
			is False. Whenever there is d_out[i]=1 and dhat[i]=0, dhat[i] is set to the value 
			of modify_zero_divide.Similarly, whenever d_out[i]=0 and dhat[i]=1, then dhat[i] is set to the
			value of modify_zero_divide.

#Available Machine Learning Methods

This section includes the options for the [Tree](#tree), [Boosted Tree](#boosted-tree),[Random Forest](#random-forest), [Ridge](#ridge regression), [RidgeLogit](#ridge-logit), [Lasso](#lasso), and [Lasso Logit](#lasso-logit)

Tree
=======
The "Tree" option uses an implementation of sklearn's [DecisionTreeRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor) class. The maximum depth of the tree is selected using sklearn's [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html). The available method_options dictionary entries are as follows:

		criterion : string, optional (default="mse")
			The function to measure the quality of a split. Supported criteria
			are "mse" for the mean squared error, which is equal to variance
			reduction as feature selection criterion, and "mae" for the mean
			 absolute error.
	
		splitter : string, optional (default="best")
			The strategy used to choose the split at each node. Supported
			strategies are "best" to choose the best split and "random" to choose
			the best random split.
		
		max_features : int, float, string or None, optional (default=None)
			the number of features to consider when looking for the best split:
			- If int, then consider `max_features` features at each split.
			- If float, then `max_features` is a percentage and
				`int(max_features * n_features)` features are considered at each split.
			- If "auto", then `max_features=n_features`.
			- If "sqrt", then `max_features=sqrt(n_features)`.
			- If "log2", then `max_features=log2(n_features)`.
			- If None, then `max_features=n_features`.
				Note: the search for a split does not stop until at least one
				valid partition of the node samples is found, even if it requires to
				effectively inspect more than ``max_features`` features.
			
		min_samples_split : int, float, optional (default=2)
			The minimum number of samples required to split an internal node:
			- If int, then consider `min_samples_split` as the minimum number.
			- If float, then `min_samples_split` is a percentage and
				`ceil(min_samples_split * n_samples)` are the minimum
				number of samples for each split.
				Added float values for percentages.
		
		min_samples_leaf : int, float, optional (default=1)
			The minimum number of samples required to be at a leaf node:
			- If int, then consider `min_samples_leaf` as the minimum number.
			- If float, then `min_samples_leaf` is a percentage and
				`ceil(min_samples_leaf * n_samples)` are the minimum
				number of samples for each node.
				Added float values for percentages.
	
		min_weight_fraction_leaf : float, optional (default=0.)
			The minimum weighted fraction of the sum total of weights (of all
			the input samples) required to be at a leaf node. Samples have
			equal weight when sample_weight is not provided.
		
		max_leaf_nodes : int or None, optional (default=None)
			Grow a tree with ``max_leaf_nodes`` in best-first fashion.
			Best nodes are defined as relative reduction in impurity.
			If None then unlimited number of leaf nodes.
		
		random_state : int, RandomState instance or None, optional (default=None)
			If int, random_state is the seed used by the random number generator;
			If RandomState instance, random_state is the random number generator;
			If None, the random number generator is the RandomState instance used
				by `np.random`.
		
		min_impurity_split : float, optional (default=1e-7)
			Threshold for early stopping in tree growth. If the impurity
			of a node is below the threshold, the node is a leaf.
			
		presort : bool, optional (default=False)
			Whether to presort the data to speed up the finding of best splits in
			fitting. For the default settings of a decision tree on large
			datasets, setting this to true may slow down the training process.
			When using either a smaller dataset or a restricted depth, this may
			speed up the training.
			
		n_jobs : int, default=1
			Number of jobs to run in parallel.
		
		cv : int, cross-validation generator or an iterable, optional (default=10)
			Determines the cross-validation splitting strategy.
			Possible inputs for cv are:
				- integer, to specify the number of folds in a `(Stratified)KFold`,
				- An object to be used as a cross-validation generator.
				- An iterable yielding train, test splits.
	
		search_range_low: int, optional. Default=1.
		search_range_high: int, optional. Default=10.
			This method uses a cross-validated grid search to estimate the optimal max_depth
			that is inclusively between search_range_low and search_range_high
				
Boosted Tree
=======
The Boosted Tree option uses sklearn's [AdaBoostRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html#sklearn.ensemble.AdaBoostRegressor) as an implementation of the boosting algorithm [AdaBoost](http://scikit-learn.org/stable/modules/ensemble.html#adaboost). The available method_options dictionary entries are as follows: 

		base_estimator : object, optional (default=DecisionTreeRegressor)
		        The base estimator from which the boosted ensemble is built.
		        Support for sample weighting is required. For options supported base estimators,
		        see sklearn's documentation of AdaBoostRegressor

		n_estimators : integer, optional (default=100)
		        The maximum number of estimators at which boosting is terminated.
		        In case of perfect fit, the learning procedure is stopped early.
	
		learning_rate : float, optional (default=0.001)
		        Learning rate shrinks the contribution of each regressor by
		        ``learning_rate``. There is a trade-off between ``learning_rate`` and
		        ``n_estimators``.
		
		loss : {'linear', 'square', 'exponential'}, optional (default='exponential')
		        The loss function to use when updating the weights after each
		        boosting iteration.
		
		random_state : int, RandomState instance or None, optional (default=None)
		        If int, random_state is the seed used by the random number generator;
		        If RandomState instance, random_state is the random number generator;
		        If None, the random number generator is the RandomState instance used
		        by `np.random`	
Random Forest
=======
The Random Forest option uses an implementation of sklearn's [RandomForestRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor) class. The available method_options dictionary entries are as follows:

	
		n_estimators : integer, optional (default=1000)
			The number of trees in the forest.
		
		criterion : string, optional (default="mse")
			The function to measure the quality of a split. Supported criteria
			are "mse" for the mean squared error, which is equal to variance
			reduction as feature selection criterion, and "mae" for the mean
			absolute error.
		max_features : int, float, string or None, optional (default='auto')
			The number of features to consider when looking for the best split:
			- If int, then consider `max_features` features at each split.
			- If float, then `max_features` is a percentage and
				int(max_features * n_features)` features are considered at each
				split.
			- If "auto", then `max_features=n_features`.
			- If "sqrt", then `max_features=sqrt(n_features)`.
			- If "log2", then `max_features=log2(n_features)`.
			- If None, then `max_features=n_features`.
				Note: the search for a split does not stop until at least one
		        	valid partition of the node samples is found, even if it requires to
		       		effectively inspect more than ``max_features`` features.
		
		max_depth : integer or None, optional (default=None)
		        The maximum depth of the tree. If None, then nodes are expanded until
		        all leaves are pure or until all leaves contain less than
		        min_samples_split samples.
		
		min_samples_split : int, float, optional (default=5)
		        The minimum number of samples required to split an internal node:
		        - If int, then consider `min_samples_split` as the minimum number.
		        - If float, then `min_samples_split` is a percentage and
		          `ceil(min_samples_split * n_samples)` are the minimum
		          number of samples for each split.
		        .. versionchanged:: 0.18
		           Added float values for percentages.
		
		min_samples_leaf : int, float, optional (default=5)
		        The minimum number of samples required to be at a leaf node:
		        - If int, then consider `min_samples_leaf` as the minimum number.
		        - If float, then `min_samples_leaf` is a percentage and
		          `ceil(min_samples_leaf * n_samples)` are the minimum
		          number of samples for each node.
		        .. versionchanged:: 0.18
		           Added float values for percentages.
		
		min_weight_fraction_leaf : float, optional (default=0.)
		        The minimum weighted fraction of the sum total of weights (of all
		        the input samples) required to be at a leaf node. Samples have
		        equal weight when sample_weight is not provided.
		
		max_leaf_nodes : int or None, optional (default=None)
		        Grow trees with ``max_leaf_nodes`` in best-first fashion.
		        Best nodes are defined as relative reduction in impurity.
		        If None then unlimited number of leaf nodes.
		
		min_impurity_split : float, optional (default=1e-7)
		        Threshold for early stopping in tree growth. A node will split
		        if its impurity is above the threshold, otherwise it is a leaf.
		        .. versionadded:: 0.18
		    bootstrap : boolean, optional (default=True)
		        Whether bootstrap samples are used when building trees.
		    oob_score : bool, optional (default=False)
		        whether to use out-of-bag samples to estimate
		        the R^2 on unseen data.
		
		n_jobs : integer, optional (default=1)
		        The number of jobs to run in parallel for both `fit` and `predict`.
		        If -1, then the number of jobs is set to the number of cores.
		
		random_state : int, RandomState instance or None, optional (default=None)
		        If int, random_state is the seed used by the random number generator;
		        If RandomState instance, random_state is the random number generator;
		        If None, the random number generator is the RandomState instance used
		        by `np.random`.
		
		verbose : int, optional (default=0)
		        Controls the verbosity of the tree building process.
		
		warm_start : bool, optional (default=False)
		        When set to ``True``, reuse the solution of the previous call to fit
		        and add more estimators to the ensemble, otherwise, just fit a whole
		        new forest.
Ridge Logit
=======
The Ridge Logit option implements the RidgeLogitCV class that is a part of the doubleML package. The L-2 penalty parameter is chosen using cross validation. The available method_options dictionary entries are as follows:

		cv: integer, optional. The number of folds used in the leave-one-out cross validation (default=10)
		
		Cs: integer or numpy array of floats, optional. If Cs is a numpy array, then the values of Cs will 
			deterimine the potential L-2 penalty parameter values that the cross validation considers. If Cs
			takes the value of an integer, then the Cs will be exponential between low_val and
			high_val(default=10)
		
		solver: Determines which solver will be used to estimate the beta values for each given C. All of the solvers
			are methods in scipy.optimize.minimize. For the solvers 'BFGS', "Newton-CG", and 'CG', an
			analytical derivative is automatically used (default='BFGS')
			Potential solver values include:
				-"BFGS"
				-"Newton-CG"
				-"CG"
				-"Nelder-Mead"
				-"Powell"
		
		solver_options: dict, optional (default=None). Options for the scipy.optimize.minimize method chosen in the
			solver method. To view the options specific to the selected solver, look at the 'options' section [here](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html)
		
		low_val: float, optional. The lowest penalty parameter value considered in cross-validation. (default=1E-3)
		
		high_val: float, optional. The  highest penalty parameter value considered in cross-validation. (default=1E3)

Ridge 
=======
The Ridge option uses an implementation of sklearn's [RidgeCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.RidgeCV) class. The L-2 penalty parameter is chosen using cross validation. The available method_options dictionary entries are as follows:

		    alphas : numpy array of shape [n_alphas], or int. Default=40.
		        -If alpha is a numpy array, then it is used as an array of L-2 penalties that are
				tested using cross validationis the array of alpha values to try.
		        -If alpha is an integer, then the computer generates L-2 penalty parameters using
				a loglinear numpy array of length alpha with values ranging from 1E-3 to 1E5
		    
		    fit_intercept : boolean
		        Whether to calculate the intercept for this model. If set
		        to false, no intercept will be used in calculations
		        (e.g. data is expected to be already centered).
		    
		    scoring : string, callable or None, optional, default: None
		        A string (see model evaluation documentation) or
		        a scorer callable object / function with signature
		        ``scorer(estimator, X, y)``.
		    
		    cv : int, cross-validation generator or an iterable, optional
		        Determines the cross-validation splitting strategy.
		        Possible inputs for cv are:
		        - None, to use the efficient Leave-One-Out cross-validation
		        - integer, to specify the number of folds.
		        - An object to be used as a cross-validation generator.
		        - An iterable yielding train/test splits.
		        For integer/None inputs, if ``y`` is binary or multiclass,
		        :class:`sklearn.model_selection.StratifiedKFold` is used, else, 
		        :class:`sklearn.model_selection.KFold` is used.
		        Refer :ref:`User Guide <cross_validation>` for the various
		        cross-validation strategies that can be used here.
		    
		    gcv_mode : {None, 'auto', 'svd', eigen'}, optional
		        Flag indicating which strategy to use when performing
		        Generalized Cross-Validation. Options are::
		            'auto' : use svd if n_samples > n_features or when X is a sparse
		                     matrix, otherwise use eigen
		            'svd' : force computation via singular value decomposition of X
		                    (does not work for sparse matrices)
		            'eigen' : force computation via eigendecomposition of X^T X
		        The 'auto' mode is the default and is intended to pick the cheaper
		        option of the two depending upon the shape and format of the training
		        data.
		    
		    store_cv_values : boolean, default=False
		        Flag indicating if the cross-validation values corresponding to
		        each alpha should be stored in the `cv_values_` attribute (see
		        below). This flag is only compatible with `cv=None` (i.e. using
		        Generalized Cross-Validation).

Lasso
=======
The Lasso option uses an implementation of sklearn's [LassoCV](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html#sklearn.linear_model.LassoCV) class. The L-1 penalty parameter is chosen using cross validation. The available method_options dictionary entries are as follows:

		
		eps : float, optional. Default=1E-3
		        Length of the path. "eps=1e-3" means that
		        "alpha_min / alpha_max = 1e-3".
		    
		n_alphas : int, optional. Default=100
		        Number of alphas along the regularization path
		
		alphas : numpy array, optional, Default=None
		        List of alphas where to compute the models. 
		        If ``None`` alphas are set automatically
		
		precompute : True | False | 'auto' | array-like. Default='auto'
		        Whether to use a precomputed Gram matrix to speed up
		        calculations. If set to 'auto' let us decide. The Gram
		        matrix can also be passed as argument.
		
		max_iter : int, optional. Default=5000
		        The maximum number of iterations
		
		tol : float, optional. Default=1E-4
		        The tolerance for the optimization: if the updates are
		        smaller than "tol", the optimization code checks the
		        dual gap for optimality and continues until it is smaller
		        than "tol".
		
		cv : int, cross-validation generator or an iterable, optional. Default=10
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
		        Number of CPUs to use during the cross validation. If "-1", use
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
		
		copy_X : boolean, optional, default True
		        If "True", X will be copied; else, it may be overwritten.
Lasso Logit
=======
		
		cv: integer, optional. The number of folds used in the leave-one-out cross validation (default=10)
		
		Cs: integer or numpy array of floats, optional. If Cs is a numpy array, then the values of Cs will deterimine 
				the potential L-1 penalty parameter values that the cross validation considers. If Cs takes the value of an
				integer, then the Cs will be exponential between low_val and high_val(default=10)
		
		solver: Determines which solver will be used to estimate the beta values for each given C. All of the solvers
				are methods in scipy.optimize.minimize. For the solvers 'BFGS', "Newton-CG", and 'CG', an
				analytical derivative is automatically used (default='BFGS')
				Potential solver values include:
					-"BFGS"
					-"Newton-CG"
					-"CG"
					-"Nelder-Mead"
					-"Powell"
		
		solver_options: dict, optional (default=None). Options for the scipy.optimize.minimize method chosen in
				the solver method.To view the options specific to the selected solver, look [here](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html)
		
		low_val: float, optional. The lowest L-1 penalty parameter value considered in cross-validation. (default=1E-4)
		
		high_val: float, optional. The  highest L-1 penalty parameter value considered in cross-validation. (default=1E4)


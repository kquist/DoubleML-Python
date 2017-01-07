#Class: RidgeLogitCV
The RidgeLogitCV class is an implementation of an estimation of a logistic regression with an L-2 penalty parameter chosen using k-fold leave-one-out cross validation
#Model Explanation
A logistic regression model is a linear model for classification also referred to as a "logit" model. In this model, there are two possible outcomes, 1 and 0. The probability that outcome "1" occurs given X and beta is ![Alt text](Logit_Generating.png?raw=true "Logit Model")
The class RidgeLogitCV uses a maximum likelihood estimator with an L-2 penalty parameter to estimate the values of beta. Notice that the log-likelihood of an outcome Y given X, and beta are as follows:
![Alt text](Logit_LL_Deriv.png?raw=true "LLV")
Thus if we set the parameter C to be the inverse of the L-2 penalty parameter, then the beta estimate is selected to minimize the following:
![Alt text](RidgeLogitMinimize.png?raw=true "RidgeLogit Minimization")
#Parameters:
	cv: integer, optional. The number of folds used in the leave-one-out cross validation (default=10)
	
	Cs: integer or numpy array of floats, optional. If Cs is a numpy array, then the values of Cs will deterimine
		the potential L-2 penalty parameter values that the cross validation considers. If Cs takes the value of an
		integer, then the Cs will be exponential between low_val and high_val(default=10)
		
	solver: Determines which solver will be used to estimate the beta values for each given C. All of the solvers are
			methods in scipy.optimize.minimize. For the solver 'SLSQP' analytical derivative is automatically used 
			(default='SlSQP')
			Potential solver values include:
				-"SLSQP"
				-"Nelder-Mead"
				-"Powell"

	solver_options: dict, optional (default=None). Options for the scipy.optimize.minimize method chosen in the solver
		method. View the options for scipy.optimize.minimize method options to view what the 
		method-specific options are.
	
	low_val: float, optional (default=1E3). The lowest L-2 penalty parameter value considered in cross-validation.
	
	high_val: float, optional (default=1E3). The  highest L-2 penalty parameter value considered in cross-validation.
	
	warm_start: boolean, optional (default=True). Whether or not warm_start is true, an OLS regression is used to set the
		initial guess of the first minimization problem that estimates beta for a given X, Y, and C. If warm_start is
		set to true, then all later estimations use the previous beta estimates as an initial guess. If warm_start is
		False, then the initial guess of all minimization problems will be calculated using OLS



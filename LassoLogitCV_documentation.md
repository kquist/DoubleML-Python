#Class: LassoLogitCV
The LassoLogitCV class is an implementation of an estimation of a logistic regression with an L-1 penalty parameter chosen using k-fold leave-one-out cross validation
#Model Explanation
A logistic regression model is a linear model for classification also referred to as a "logit" model. In this model, there are two possible outcomes, 1 and 0. The probability that outcome "1" occurs given X and beta is as follows:

![Alt text](Logit_Generating.png?raw=true "Logit Model")
The class RidgeLogitCV uses a maximum likelihood estimator with an L-1 penalty parameter to estimate the values of beta. Notice that the log-likelihood of an outcome Y given X, and beta are as follows:

![Alt text](Logit_LL_Deriv.png?raw=true "LLV")
Thus if we set the parameter C to be the inverse of the L-2 penalty parameter, then the beta estimate is selected to minimize the following:

![Alt text](LassoLogitMinimize.png?raw=true "LassoLogit Minimization")
#Parameters:

- cv: integer, optional. The number of folds used in the leave-one-out cross validation (default=10)
	
- Cs: integer or numpy array of floats, optional. If Cs is a numpy array, then the values of Cs will deterimine the potential L-1 penalty parameter values that the cross validation considers. If Cs takes the value of an integer, then the Cs will be exponential between low_val and high_val(default=10)
		
- solver: Determines which solver will be used to estimate the beta values for each given C. All of the solvers are methods in scipy.optimize.minimize. For the solver 'SLSQP' analytical derivative is automatically used. (default='SlSQP')
	
		Potential solver values include:
		-"SLSQP"
		-"Nelder-Mead"
		-"Powell"

- solver_options: dict, optional. Options for the scipy.optimize.minimize method chosen in the solver method. If solver='SLSQP' then solver_options will default to {"maxiter": 1000, "ftol":1E-5}. If the solver is "Nelder-Mead" or "Powell" then solver_options will default to None. Click [here](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.optimize.minimize.html) to view what the method-specific options are.
	
- low_val: float, optional (default=1E3). The lowest L-1 penalty parameter value considered in cross-validation.
	
- high_val: float, optional (default=1E3). The  highest L-1 penalty parameter value considered in cross-validation.
	
- warm_start: boolean, optional (default=True). Whether or not warm_start is true, an OLS regression is used to set the initial guess of the first minimization problem that estimates beta for a given X, Y, and C. If warm_start is set to true, then all later estimations use the previous beta estimates as an initial guess. If warm_start is False, then the initial guess of all minimization problems will be calculated using OLS
#Attributes

- coefficients: numpy array of the model's estimated coefficients. coefficients[-1] is the intercept and coefficients[i] corresponds to the ith regresssor in the model

#Methods:

- fit(self,X,Y): employs k-fold cross validation to determine which L-1 penalty parameter should be used using a maximum likelihood loss function, and then estimates the coefficients of a logistic regression on the whole set when using the calculated L-2 penalty parameter. The results are stored in self.coefficients
				
		X: mxn array where there are m observations in the sample and n regressors.
		Y: 1-d array of length m where there are m observations, represents the outcome.
	
- predict(self,X): once the LassoLogitCV class has been fitted, the predict function uses these coefficients to predict the probability of each X having an outcome variable of 1
		
		X: mxn array where there are m observations in the sample and n regressors.

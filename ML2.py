import numpy as np
import warnings
import math
from scipy.optimize import minimize
import sklearn
from sklearn import tree
from sklearn import linear_model
from sklearn import ensemble
from sklearn import neural_network
from sklearn.model_selection import KFold
import statsmodels.api as sm
import warnings
import time

"""
	The DoubleML package is intended to be used to implement the estimation procedure developed in `Double Machine 
	Learning for Treatment and Causal Parameters' by Victor Chernozhukov, Denis Chetverikov, Mert Demirer,
	Esther Duflo, Christian Hansen, and Whitney Newey. 

	The package contains 3 classes: ML2Estimator, LassoLogitCV, and RidgeLogitCV.

	The ML2Estimator class implements both the partial linear estimator and the interactive estimator outlined in Chernozhukov et. al.
	This class can use a variety of machine learning  echniques (Regression Trees, Ada Boosted Trees, Random Forest, Lasso,
	Ridge, and Logistic regression with either an L-1 penalty parameter or an L-2 penalty parameter).

	The LassoLogitCV class is an implementation of a logistic regression with an L-1 penalty parameter chosen by leave-one-out cross validatoion

	The RidgeLogitCV class is an implementation of a logistic regression with an L-2 penatly parameter chosen by leave-one-out cross validation
"""

class LassoLogitCV:
	"""
	This class is a regression model with a logistic loss function that has an L-1 penatly parameter. The 'fit'
	class estimates this model and penalty parameter using leave-one-out cross validation.

	Parameters
    ----------
	cv: integer, optional. The number of folds used in the leave-one-out cross validation (default=10)
	Cs: integer or numpy array of floats, optional. If Cs is a numpy array, then the values of Cs will deterimine 
		the potential L-1 penalty parameter values that the cross validation considers. If Cs takes the value of an
		integer, then the Cs will be exponential between low_val and high_val(default=10)
	solver: Determines which solver will be used to estimate the beta values for each given C. All of the solvers
		are methods in scipy.optimize.minimize. For the solvers 'BFGS', "Newton-CG", and 'CG', an
		analytical derivative is automatically used (default='BFGS')
		Potential solver values include:
			-"SLSQP"
			-"Nelder-Mead"
			-"Powell"
	solver_options: dict, optional. Options for the scipy.optimize.minimize method chosen in the solver method.
		View the options for scipy.optimize.minimize method options to view what the method-specific options are.
		If solver='SLSQP' then solver_options will default to {"maxiter": 1000, "ftol":1E-5}.
		If the solver is "Nelder-Mead" or "Powell" then solver_options will default to None.
	low_val: float, optional. The lowest L-1 penalty parameter value considered in cross-validation. (default=1E-4)
	high_val: float, optional. The  highest L-1 penalty parameter value considered in cross-validation. (default=1E4)
		warm_start: boolean, optional (default=True). Whether or not warm_start is true, an OLS regression is used to set
		the initial guess of the first minimization problem that estimates beta for a given X, Y, and C. If warm_start
		is set to true, then all later estimations use the previous beta estimates as an initial guess. If warm_start is
		False, then the initial guess of all minimization problems will be calculated using OLS
	
	Attributes
	----------
	coefficients: numpy array of the model's estimated coefficients. coef[-1] is the intercept and coef[i] corresponds to the ith
		regresssor in the model
	
	Methods
	----------
	fit(self,X,Y): employs k-fold cross validation to determine which L-1 penalty parameter should be used using a maximum likelihood loss function,
	    and then estimates the coefficients of a logistic regression on the whole set when using the calculated L-1 penalty parameter. The results
		are stored in self.coefficients
	predict(self,X): once the LassoLogitCV class has been fitted, the predict function uses these coefficients to predict the probability of each X
		having an outcome variable of 1
	Private Methods
	----------
	_llv(self,beta,X,Y): returns the log-likelihood of the outcome Y given a logistic regression generating model with regressors X and coefficients beta.
		This is the loss function used to estimate the optimal L-1 penatly parameter
	_min_func (self,beta,C,X,Y): the loss function used to estimate beta for a given C, X, and Y.
	_d_min_func(self,beta,C,X,Y): the derivative of min_func with respect to beta. It is used 
	_calc_beta(self,X,Y,C,beta_guess=None): estimate the value of beta for a given X,Y,C

	"""
	def __init__ (self,cv=10, Cs=10,solver='SLSQP',solver_options=None, low_val=1E-3, high_val=1E3,warm_start=True):
		if isinstance(Cs,int):
			self.lambdas=np.exp(np.linspace(math.log(low_val),math.log(high_val),Cs))
		else:
			self.lambdas=Cs
		self.coefficients=None
		self.cv=cv
		self.solver=solver
		self.solver_options=solver_options
		if self.solver=='SLSQP':
			if self.solver_options is None:
				self.solver_options= {"maxiter": 1000, "ftol":1E-5}
		self.warm_start=warm_start
	def fit(self,X,y):
		init_guess_model=linear_model.LinearRegression().fit(X,y)
		init_guess=np.concatenate((init_guess_model.coef_,np.array([0.])))
		kf = KFold(n_splits=self.cv,shuffle=True)
		lambda_log_likes=np.zeros(len(self.lambdas))
		for i in range(len(self.lambdas)):
			for train, test in kf.split(X):
				start=time.time()
				index=np.zeros(len(X))
				for j in range(len(train)):
					index[train[j]]=1
				y_train=y[index==1]
				x_train=X[index==1]
				y_test=y[index==0]
				x_test=X[index==0]
				beta=self._calc_beta(x_train, y_train, self.lambdas[i],beta_guess=init_guess)
				ll=self._llv(beta,x_test,y_test)
				lambda_log_likes[i]+=ll
				if self.warm_start:
					init_guess=beta
		lambda_calc=self.lambdas[np.argmax(lambda_log_likes)]
		coef=self._calc_beta(X,y,lambda_calc,beta_guess=init_guess)
		self.coefficients=coef
		return self
	def predict(self,X):
		if self.coefficients is None:
			raise NameError("This instance of the LassoLogitCV class must be fitted before it can be used to predict")
		a=1./(1+np.exp(np.dot(X,np.array([-1*self.coefficients[:-1]]).T)-self.coefficients[-1]))
		return a.T[0]
	def _llv(self,beta,X,Y):
		"""
		This method computes the log likelihood value of Y given a logistic regression model with coefficients beta and regressors X
		
		Parameters
		----------
		X: mxn numpy array of floats, where m is the number of observations and n is the number of regressors
		Y: 1-d numpy array of length m, where each entry is either 1 or 0. Y is the outcome variable 
		beta: 1-d array of floats with length m.  beta[-1] is the intercept
			and for all i>0 beta[i] corresponds to X[:,i]
		"""
		ones=np.ones((len(X),1))
		combo_X=np.concatenate((X,ones),axis=1)
		projection=-np.dot(combo_X,np.array([beta]).T)
		return (-np.dot(Y,projection)+np.sum(projection-np.log(np.exp(projection)+1)))[0]
	def _min_func(self,beta,C,X,Y):
		"""
		This method is the function minimized to calculate beta
		
		Parameters
		----------
		X: mxn numpy array of floats, where m is the number of observations and n is the number of regressors
		Y: 1-d numpy array of length m, where each entry is either 1 or 0. Y is the outcome variable 
		C: Inverse of L-1 penalty parameter. A high C corresponds to little flexibility with betas.
		beta: 1-d array of floats with length m.  beta[-1] is the intercept
			and for all i>0 beta[i] corresponds to X[:,i]
		"""
		#print(C)
		return np.sum(abs(beta[:-1]))-C*self._llv(beta,X,Y)
		#return -C*self._llv(beta,X,Y)
	def _d_min_func(self,beta,C,X,Y):
		"""
		This method is calculates the derivative of lasso_min with respect to beta
		
		Parameters
		----------
		X: mxn numpy array of floats, where m is the number of observations and n is the number of regressors
		Y: 1-d numpy array of length m, where each entry is either 1 or 0. Y is the outcome variable 
		C: Inverse of L-1 penalty parameter. A high C corresponds to little flexibility with betas.
		beta: 1-d array of floats with length m.  beta[-1] is the intercept
			and for all i>0 beta[i] corresponds to X[:,i]
		"""
		ones=np.ones((len(X),1))
		combo_X=np.concatenate((X,ones),axis=1)
		projection=-1*(np.dot(combo_X,np.array([beta]).T)).T[0]
		d_factor=np.array([Y])-1+np.array([np.exp(projection)/(1+np.exp(projection))])
		#beta_0_d=-C*np.array([np.sum(d_factor)])
		fit_factor=-C*np.dot(d_factor,combo_X)[0]
		L1_factor=np.concatenate(((1*(beta[:-1]>0)-(1*beta[:-1]<0),np.array([0.]))))
		#beta_1_d=(1*(beta[:-1]>0)-(1*beta[:-1]<0)-C*np.dot(d_factor,X))[0]
		return fit_factor+L1_factor


	def _calc_beta(self,X,Y,C,beta_guess=None):
		"""
		This method estimates the value of beta that minimizes min_func.
		Parameters
		----------
		X: mxn numpy array of floats, where m is the number of observations and n is the number of regressors
		Y: 1-d numpy array of length m, where each entry is either 1 or 0. Y is the outcome variable 
		C: Inverse of L-1 penalty parameter. A high C corresponds to little flexibility with betas.
		beta_guess: 1-d array of floats with length m, or None. beta_guess is the initial guess that the minimizing function
			uses for beta. If beta_guess is None, then an OLS regression will be used to compute beta_guess.
			The last value of beta_guess is the intercept and for all i>0 beta_guess[i] corresponds to X[:,i]
		"""
		if beta_guess is None:
			beta_guess=np.concatenate((linear_model.LinearRegression().fit(X,Y).coef_,np.array([0.])))
		loc_min_func=lambda bet: self._min_func(bet,C,X,Y)
		#print(self.solver)
		if self.solver=='SLSQP':
			#print("right track")
			der_llv=lambda bet: self._d_min_func(bet,C,X,Y)
			answer=minimize(loc_min_func,beta_guess,method='SLSQP',jac=der_llv,options=self.solver_options)
			if answer.success:
				return answer.x
			else:
				answer_2=minimize(loc_min_func,answer.x,method='Powell')
				if answer_2.success:
					return answer_2.x
				else:
					answer_3=minimize(loc_min_func,answer_2.x,method='Nelder-Mead')
					if answer_3.success:
						return answer_3.x
					else:
						warnings.warn("Warning: optimizers were unable to converge ")

			
			
		else:
			if self.solver=="Powell":
				alternative_solver="Nelder-Mead"
			else:
				alternative_solver="Powell"
			der_llv=lambda bet: self._d_min_func(bet,C,X,Y)
			answer=minimize(loc_min_func,beta_guess,method=self.solver,options=self.solver_options)
			
			if answer.success:
				return answer.x
			else:
				answer_2=minimize(loc_min_func,answer.x,method="SLSQP",jac=der_llv,options={"maxiter":500, "ftol":1E-5})
				if answer_2.success:
					return answer_2.x
				else:
					answer_3=minimize(loc_min_func,answer_2.x,method=alternative_solver)
					if answer_3.success:
						return answer_3.x
					else:
						warnings.warn("Warning: optimizers were unable to converge")






class RidgeLogitCV:
	"""
	This model solves a regression model with a logistic loss function and
	and a penalty of the L-2 norm of the betastercept betas. Leave-one-out cross validation is used
	to select the penalty parameter for the L-2 norm

	parameters
    ----------
	cv: integer, optional. The number of folds used in the leave-one-out cross validation (default=10)
	Cs: integer or numpy array of floats, optional. If Cs is a numpy array, then the values of Cs will deterimine 
		the potential L-2 penalty parameter values that the cross validation considers. If Cs takes the value of an
		integer, then the Cs will be exponential between low_val and high_val(default=10)
	solver: Determines which solver will be used to estimate the beta values for each given C. All of the solvers
		are methods in scipy.optimize.minimize. For the solver 'SLSQP', an
		analytical derivative is automatically used (default='SLSQP')
		Potential solver values include:
			-"SLSQP"
			-"Nelder-Mead"
			-"Powell"
	solver_options: dict, optional. Options for the scipy.optimize.minimize method chosen in the solver method.
		View the options for scipy.optimize.minimize method options to view what the method-specific options are.
		If solver='SLSQP' then solver_options will default to {"maxiter": 1000, "ftol":1E-5}.
		If the solver is "Nelder-Mead" or "Powell" then solver_options will default to None.
	low_val: float, optional (default=1E3). The lowest L-2 penalty parameter value considered in cross-validation.
	high_val: float, optional (default=1E3). The  highest L-2 penalty parameter value considered in cross-validation.
	warm_start: boolean, optional (default=True). Whether or not warm_start is true, an OLS regression is used to set
		the initial guess of the first minimization problem that estimates beta for a given X, Y, and C. If warm_start
		is set to true, then all later estimations use the previous beta estimates as an initial guess. If warm_start is
		False, then the initial guess of all minimization problems will be calculated using OLS

	Attributes
	----------
	coefficients: numpy array of the model's estimated coefficients. coef[-1] is the intercept and coef[i] corresponds to the ith
		regresssor in the model
	
	Methods
	----------
	fit(self,X,Y): employs k-fold cross validation to determine which L-1 penalty parameter should be used using a maximum likelihood loss function,
	    and then estimates the coefficients of a logistic regression on the whole set when using the calculated L-2 penalty parameter. The results
		are stored in self.coefficients
	predict(self,X): once the LassoLogitCV class has been fitted, the predict function uses these coefficients to predict the probability of each X
		having an outcome variable of 1
	
	Private Methods
	----------
	_llv(self,beta,X,Y): returns the log-likelihood of the outcome Y given a logistic regression generating model with regressors X and coefficients beta.
		This is the loss function used to estimate the optimal L-2 penatly parameter
	_min_func (self,beta,C,X,Y): the loss function used to estimate beta for a given C, X, and Y.
	_d_min_func(self,beta,C,X,Y): the derivative of min_func with respect to beta. It is used 
	_calc_beta(self,X,Y,C,beta_guess=None):_calc_beta(self,X,Y,C,beta_guess=None): estimate the value of beta for a given X,Y,C
	"""
	def __init__(self,cv=10, Cs=10,solver='SLSQP',solver_options=None, low_val=1E-3, high_val=1E3,warm_start=True):
		if isinstance(Cs,int):
			self.lambdas=np.exp(np.linspace(math.log(low_val),math.log(high_val),Cs))
		else:
			self.lambdas=Cs
		self.coefficients=None
		self.cv=cv
		self.solver=solver
		self.solver_options=solver_options
		if self.solver=='SLSQP':
			if self.solver_options is None:
				self.solver_options= {"maxiter": 1000, "ftol":1E-5}
		self.warm_start=warm_start
	def fit(self,X,y):
		init_guess_model=linear_model.LinearRegression().fit(X,y)
		init_guess=np.concatenate((init_guess_model.coef_,np.array([0.])))
		kf = KFold(n_splits=self.cv,shuffle=True)
		lambda_log_likes=np.zeros(len(self.lambdas))
		for i in range(len(self.lambdas)):
			for train, test in kf.split(X):
				start=time.time()
				index=np.zeros(len(X))
				for j in range(len(train)):
					index[train[j]]=1
				y_train=y[index==1]
				x_train=X[index==1]
				y_test=y[index==0]
				x_test=X[index==0]
				beta=self._calc_beta(x_train, y_train, self.lambdas[i],beta_guess=init_guess)
				ll=self._llv(beta,x_test,y_test)
				lambda_log_likes[i]+=ll
				if self.warm_start:
					init_guess=beta
		lambda_calc=self.lambdas[np.argmax(lambda_log_likes)]
		coef=self._calc_beta(X,y,lambda_calc,beta_guess=init_guess)
		self.coefficients=coef
		return self
	def predict(self,X):
		if self.coefficients is None:
			raise NameError("This instance of the RidgeLogitCV class must be fitted before it can be used to predict")
		a=1./(1+np.exp(np.dot(X,np.array([-1*self.coefficients[:-1]]).T)-self.coefficients[-1]))
		return a.T[0]
	def _llv(self,beta,X,Y):
		"""
		This method computes the log likelihood value of the logistic regression
		
		Parameters
		----------
		X: mxn numpy array of floats, where m is the number of observations and n is the number of regressors
		Y: 1-d numpy array of length m, where each entry is either 1 or 0. Y is the outcome variable 
		C: Inverse of L-2 penalty parameter. A high C corresponds to little flexibility with betas.
		beta_guess: 1-d array of length m, or None. beta_guess is the initial guess that the minimizing function
			uses for beta. If beta_guess is None, then an OLS regression will be used to compute beta_guess.
			The last value of beta_guess is the intercept and for all i>0 beta_guess[i] corresponds to X[:,i-1]
		"""
		ones=np.ones((len(X),1))
		combo_X=np.concatenate((X,ones),axis=1)
		projection=-np.dot(combo_X,np.array([beta]).T)
		return (-np.dot(Y,projection)+np.sum(projection-np.log(np.exp(projection)+1)))[0]
	def _min_func(self,beta,C,X,Y):
		"""
		This function is minimized to estimate beta for a given C, X, Y
		
		Parameters
		----------
		X: mxn numpy array of floats, where m is the number of observations and n is the number of regressors
		Y: 1-d numpy array of length m, where each entry is either 1 or 0. Y is the outcome variable 
		C: Inverse of L-2 penalty parameter. A high C corresponds to little flexibility with betas.
		beta_guess: 1-d array of length m, or None. beta_guess is the initial guess that the minimizing function
			uses for beta. If beta_guess is None, then an OLS regression will be used to compute beta_guess.
			The last value of beta_guess is the intercept and for all i>0 beta_guess[i] corresponds to X[:,i]
		"""
		return np.sqrt(np.sum(beta[:-1]**2))-C*self._llv(beta,X,Y)


	def _d_min_func(self,beta,C,X,Y):
		"""
		This method is the derivative of min_func with respect to beta, and is used to estimate beta

		Parameters
		----------
		Y: Outcome variable
		X: matrix of regressors
		Beta: beta[-1] is the intercept, beta[i] corressponds to X[i] for all i>0
		C: Inverse of L-2 penalty parameter. A high C corresponds to little flexibility with betas.
		"""
		ones=np.ones((len(X),1))
		combo_X=np.concatenate((X,ones),axis=1)
		projection=-1*(np.dot(combo_X,np.array([beta]).T)).T[0]
		d_factor=np.array([Y])-1+np.array([np.exp(projection)/(1+np.exp(projection))])
		#beta_0_d=-C*np.array([np.sum(d_factor)])
		fit_factor=-C*np.dot(d_factor,combo_X)[0]
		L2_factor=np.concatenate(((beta[:-1]/np.sqrt(np.sum(beta[:-1]**2)),np.array([0.]))))
		#beta_1_d=(1*(beta[:-1]>0)-(1*beta[:-1]<0)-C*np.dot(d_factor,X))[0]
		return fit_factor+L2_factor


	def _calc_beta(self,X,Y,C,beta_guess=None):
		"""
		This method calculates the value of beta that minimizes min_func
		Parameters
		----------
		X: mxn numpy array of floats, where m is the number of observations and n is the number of regressors
		Y: 1-d numpy array of length m, where each entry is either 1 or 0. Y is the outcome variable 
		C: Inverse of L-2 penalty parameter. A high C corresponds to little flexibility with betas.
		beta_guess: 1-d array of length m, or None. beta_guess is the initial guess that the minimizing function
			uses for beta. If beta_guess is None, then an OLS regression will be used to compute beta_guess.
			The last value of beta_guess is the intercept and for all i>0 beta_guess[i] corresponds to X[:,i]

		"""
		if beta_guess is None:
			beta_guess=np.concatenate((linear_model.LinearRegression().fit(X,Y).coef_,np.array([0.])))
		loc_min_func=lambda bet: self._min_func(bet,C,X,Y)
		#print(self.solver)
		if self.solver=='SLSQP':
			#print("right track")
			der_llv=lambda bet: self._d_min_func(bet,C,X,Y)
			answer=minimize(loc_min_func,beta_guess,method='SLSQP',jac=der_llv,options=self.solver_options)
			if answer.success:
				return answer.x
			else:
				answer_2=minimize(loc_min_func,answer.x,method='Powell')
				if answer_2.success:
					return answer_2.x
				else:
					answer_3=minimize(loc_min_func,answer_2.x,method='Nelder-Mead')
					if answer_3.success:
						return answer_3.x
					else:
						warnings.warn("Warning: optimizers were unable to converge")

			
			
		else:
			if self.solver=="Powell":
				alternative_solver="Nelder-Mead"
			else:
				alternative_solver="Powell"
			der_llv=lambda bet: self._d_min_func(bet,C,X,Y)
			answer=minimize(loc_min_func,beta_guess,method=self.solver,options=self.solver_options)
			
			if answer.success:
				return answer.x
			else:
				answer_2=minimize(loc_min_func,answer.x,method="SLSQP",jac=der_llv,options={"maxiter":500, "ftol":1E-5})
				if answer_2.success:
					return answer_2.x
				else:
					answer_3=minimize(loc_min_func,answer_2.x,method=alternative_solver)
					if answer_3.success:
						return answer_3.x
					else:
						warnings.warn("Warning: optimizers were unable to converge")


class ML2Estimator:
	"""
	This class is an implementation of the estimation procedure developed in `Double Machine 
	Learning for Treatment and Causal Parameters' by Victor Chernozhukov, Denis Chetverikov, Mert Demirer,
	Esther Duflo, Christian Hansen, and Whitney Newey
    

	Parameters
    ----------
	
	method: str, optional (default="Tree")
		ML estimation technique to be used when the outcome variable is continuous
			options:
			-"Tree" regression tree
			-"Lasso"
			-"Ridge"
			-"Random Forest"
			-"Boosted Tree" Ada Boosted Tree 
	method_binary: str, or none, optional.
		ML estimation technique to be used when the outcome variable is continuous. The predicted
		values represent probabilities
		The default value is the value of `method' unless method is `Lasso' or `Ridge'
		in which case the default value is `Lasso Logit' or `Ridge Logit' respectively. 
		options:
			-"Tree" regression tree
			-"Lasso Logit"
			-"Lasso"
			-"Ridge Logit"
			-"Ridge"
			-"Random Forest"
			-"Boosted Tree" Ada Boosted Tree 
		
	method_options: dict, or none, optional
		Options for the ml estimation technique for continuous outcome variable.
		Default values depend on machine learning method.
		
		---Boosted Tree Method Options---

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



		---Lasso Method Options---

			eps : float, optional. Default=1E-3
		        Length of the path. ``eps=1e-3`` means that
		        ``alpha_min / alpha_max = 1e-3``.
		    n_alphas : int, optional. Default=100
		        Number of alphas along the regularization path
		    alphas : numpy array, optional, Default=None
		        List of alphas where to compute the models. 
		        If ``None`` alphas are set automatically
		    precompute : True | False | 'auto' | array-like. Default='auto'
		        Whether to use a precomputed Gram matrix to speed up
		        calculations. If set to ``'auto'`` let us decide. The Gram
		        matrix can also be passed as argument.
		    max_iter : int, optional. Default=5000
		        The maximum number of iterations
		    tol : float, optional. Default=1E-4
		        The tolerance for the optimization: if the updates are
		        smaller than ``tol``, the optimization code checks the
		        dual gap for optimality and continues until it is smaller
		        than ``tol``.
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
		    copy_X : boolean, optional, default True
		        If ``True``, X will be copied; else, it may be overwritten.
		


		---Lasso Logit Options---  
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
			solver_options: dict, optional (default=None). Options for the scipy.optimize.minimize method chosen in the solver method.
				View the options for scipy.optimize.minimize method options to view what the method-specific options are.
			low_val: float, optional. The lowest L-1 penalty parameter value considered in cross-validation. (default=1E-4)
			high_val: float, optional. The  highest L-1 penalty parameter value considered in cross-validation. (default=1E4)


		----Random Forest Method options---
	
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





		---Ridge Method Options---

			alphas : numpy array of shape [n_alphas], or int. Default=40.
		        -If it is a numpy array, then it is the array of alpha values to try.
		        	Regularization strength; must be a positive float. Regularization
		        	improves the conditioning of the problem and reduces the variance of
		        	the estimates. Larger values specify stronger regularization.
		        	Alpha corresponds to ``C^-1`` in other linear models such as 
		        	LogisticRegression or LinearSVC.
		        -If alpha is an integer, then a loglinear numpy array of length alpha is created
		        	with values ranging from 1E-3 to 1E3
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




    	---Ridge Logit Options---
    
		    cv: integer, optional. The number of folds used in the leave-one-out cross validation (default=10)
			Cs: integer or numpy array of floats, optional. If Cs is a numpy array, then the values of Cs will deterimine 
				the potential L-2 penalty parameter values that the cross validation considers. If Cs takes the value of an
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
			solver_options: dict, optional (default=None). Options for the scipy.optimize.minimize method chosen in the solver method.
				View the options for scipy.optimize.minimize method options to view what the method-specific options are.
			low_val: float, optional. The lowest L-2 penalty parameter value considered in cross-validation. (default=1E-3)
			high_val: float, optional. The  highest L-2 penalty parameter value considered in cross-validation. (default=1E3)



		----Tree Method options---

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
				.. versionadded:: 0.18
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
	


	Attributes
	----------
		method_class: The initialized instance of the class of mahcine learning method that self.method corresponds to.
			For example, if 'self.method'="Random Forest", then 'self.method_class' will be an instance of a Random Forest estimator
			with the settings defined by 'method_options'.

		method_class_binary: The initialized instance of the class of mahcine learning method that self.method_binary corresponds to.
			For example, if 'self.method_binary'="Ridge Logit", then 'self.method_class_binary' will be an instance of a Random Forest estimator
			with the settings defined by 'method_options_binary'.

		pl_beta: The effect of d on the outcome variable using the partial linear estimation strategy

		pl_se: The standard error of the effect of d on the outcome variable using the partial linear estimation strategy:

		interactive_beta: The effect of d on the outcome variable using the partial linear estimation strategy

		interactive_se: The standard error of the effect of d on the outcome variable using the partial linear estimation strategy:


	Methods
	----------
		fit(self,X,Y,binary_outcome): This method fits a machine learning method using regressors X and the outcome variable Y. If binary_outcome is True,
			then self.method_class_binary will be the ml method used. If binary_outcome is False, then self.method_class will be used. X should be an mxn numpy array
			whwere m is the number of observations and n are the regressors. Y should be 1-d numpy array of length m
		
		pl_Estimate(self,X,y,d,test_size, normalize,second_order_terms verbose, standard_errors):: This method is the implementation of the double machine learning
			partial linear estimation explained in Chernozhukov et. al. This method returns the class with the beta estimate stored in self.PL_beta and the standard error
			stored in self.PL_se

		interactive_Estimate(self,X,y,d,test_size,normalize, second_order_terms, drop_zero_divide, modify_zero_divide,verbose): This method is the implementation of the double machine learning
			interactive estimation explained in Chernozhukov et. al. This method returns the class with the beta estimate stored in self.Interactive_beta and the standard error
			stored in self.Interactive_se

	Private Methods
	-----------
		_define_lasso(self,binary_outcome): Initializes and returns an instance of the sklearn.linear_model.LassoCV class. If binary_outcome is True,
			then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

		_define_lasso_logit(self,binary_outcome): Initializes and returns  an instance of the LassoLogitCV class.If binary_outcome is True,
			then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

		_define_random_forest(self,binary_outcome): Initializes and returns an instance of the sklearn.ensemble.RandomForestRegressor class.If binary_outcome is True,
			then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

		_define_regression_tree(self,binary_outcome): Initializes and returns  an instance of sklearn.model_selection.GridSearchCV() class where
			where tree.DecisionTreeRegressor is the model being searched over.If binary_outcome is True,
			then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

		_define_regression_tree_boosted(self,binary_outcome): Initializes and returns  an instance of the sklearn.ensemble.AdaBoostRegressor class.If binary_outcome is True,
			then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

		_define_ridge(self,binary_outcome): Initializes and returns  an instance of the sklearn.linear_model.RidgeCV class. If binary_outcome is True,
			then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

		_define_ridge_logit(self,binary_outcome): Initializes and returns an instance of the RidgeLogitCV class. If binary_outcome is True,
			then method_options_binary will be used. If binary_outcome is False, method_options will be used. 

		_define_model(self,binary_outcome): If binary_outcome is  false, this method will initialize and return a member of class type self.method with self.method_options
			If binary_outcome is False, this method will initialize and return a member of class type self.method_binary with self.method_options_binary
		_find_residuals(self, y_use,y_out,x_use,x_out,binary_outcome): This method uses x_use and y_use to fit a model, and then uses that calculates yhat=E[y|x_out].
			This method returns yhat and its residuals (y_out-yhat)
		_normalize(self,X): divides each regressor of X by its standard deviation and returns the resulting matrix
		_so_terms(self,X): prodcues the second order terms of a matrix X, and returns both the first and second order terms
	"""
	def __init__(self, method="Tree", method_binary=None, method_options=None, method_options_binary=None):
		self.method=method
		self.method_options=method_options
		if method_binary is None:
			if method=="Lasso":
				self.method_binary="Lasso Logit"
			elif method=="Ridge":
				self.method_binary="Ridge Logit"
			else:
				self.method_binary=self.method
		else:
			self.method_binary=method_binary
		self.method_options_binary=method_options_binary
		if method_options_binary is None:
			if self.method==self.method_binary:
				self.method_options_binary=self.method_options
		self.method_class=self._define_model(binary_outcome=False)
		self.method_class_binary=self._define_model(binary_outcome=True)
		self.pl_beta=None
		self.pl_se=None
		self.interactive_beta=None
		self.interactive_se=None

	def _define_lasso(self,binary_outcome):
		if binary_outcome:
			m_options=self.method_options_binary
		else:
			m_options=self.method_options
		lasso_options={"eps":0.001, "cv":10,"n_alphas":100,"alphas":None,"fit_intercept":True, "precompute":'auto',
			"max_iter":5000, "tol":0.0001, "copy_X":True, "verbose":False, "n_jobs":1, "positive":False, "random_state":None,
			"selection":'cyclic'}
		if m_options!=None:
			if isinstance(m_options,dict):
				for i in m_options:
					if i in lasso_options:
						lasso_options[i]=m_options[i]
					else:
						print("Error: ", i," is not a valid option entry")
						print("valid option entries include:")
						for i in lasso_options:
							print(i)
						raise NameError("Invalid option entry")		
			else:
				raise NameError("options must take the form of a dictionary")
		#print (lasso_options)
		eps=lasso_options["eps"]
		cv=lasso_options["cv"]
		n_alphas=lasso_options["n_alphas"]
		alphas=lasso_options["alphas"]
		fit_intercept=lasso_options["fit_intercept"]
		precompute=lasso_options["precompute"]
		max_iter=lasso_options["max_iter"]
		tol=lasso_options["tol"]
		copy_X=lasso_options["copy_X"]
		verbose=lasso_options["verbose"]
		n_jobs=lasso_options["n_jobs"]
		positive=lasso_options["positive"]
		random_state=lasso_options["random_state"]
		selection=lasso_options["selection"]
		return linear_model.LassoCV(cv=cv, eps=eps, fit_intercept=fit_intercept, n_alphas=n_alphas, alphas=alphas,
				precompute=precompute,max_iter=max_iter,tol=tol,copy_X=copy_X,verbose=verbose, n_jobs=n_jobs, positive=positive,
				random_state=random_state,selection=selection)

	def _define_lasso_logit(self,binary_outcome):
		ll_options={"cv": 10, "Cs": 10, "solver":'SLSQP', "solver_options":None,"low_val": 1E-3, "high_val":1E3}
		m_options=self.method_options_binary
		if m_options!=None:
			if isinstance(m_options,dict):
				for i in m_options:
					if i in ll_options:
						ll_options[i]=m_options[i]
					else:
						print("Error: ",i," is not a valid option entry")
						print("valid option entries include:")
						for i in ll_options:
							print(i)
						raise NameError("Invalid option entry")	
						raise NameError("Invalid option entry")		
			else:
				raise NameError("options must take the form of a dictionary")
		#print (ll_options)
		cv=ll_options["cv"]
		Cs=ll_options["Cs"]
		solver=ll_options["solver"]
		solver_options=ll_options["solver_options"]
		low_val=ll_options["low_val"]
		high_val=ll_options["high_val"]
		return LassoLogitCV(cv=cv, Cs=Cs, solver=solver, low_val=low_val, high_val=high_val)

	def _define_random_forest(self,binary_outcome):
		if binary_outcome:
			m_options=self.method_options_binary
		else:
			m_options=self.method_options
		rf_options={"n_estimators": 1000, "criterion": 'mse', "max_depth": None, "min_samples_split": 5, "min_samples_leaf": 5, 
			"min_weight_fraction_leaf":0.0, "max_features":"auto", "max_leaf_nodes":None, "min_impurity_split":1e-7, 
			"bootstrap":True, "oob_score":False, "n_jobs":1, "random_state":None, "verbose":0, "warm_start":False}
		if m_options!=None:
			if isinstance(m_options,dict):
				for i in m_options:
					if i in rf_options:
						rf_options[i]=m_options[i]
					else:
						print("Error: ",i," is not a valid option entry")
						print("valid option entries include:")
						for i in rf_options:
							print(i)
						raise NameError("Invalid option entry")	
						raise NameError("Invalid option entry")		
			else:
				raise NameError("options must take the form of a dictionary")
		#print (rf_options)
		n_estimators=rf_options["n_estimators"]
		criterion=rf_options["criterion"]
		max_depth=rf_options["max_depth"]
		min_samples_split=rf_options["min_samples_split"]
		min_samples_leaf=rf_options["min_samples_leaf"]
		min_weight_fraction_leaf=rf_options["min_weight_fraction_leaf"]
		max_features=rf_options["max_features"]
		max_leaf_nodes=rf_options["max_leaf_nodes"]
		min_impurity_split=rf_options["min_impurity_split"]
		bootstrap=rf_options["bootstrap"]
		oob_score=rf_options["oob_score"]
		n_jobs=rf_options["n_jobs"]
		random_state=rf_options["random_state"]
		verbose=rf_options["verbose"]
		warm_start=rf_options["warm_start"]
		return ensemble.RandomForestRegressor(n_estimators=n_estimators,criterion=criterion,max_depth=max_depth,
			min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf,
			max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_split=min_impurity_split, bootstrap=bootstrap,
			oob_score=oob_score,n_jobs=n_jobs, random_state=random_state, verbose=verbose)

	def _define_regression_tree(self,binary_outcome):
		if binary_outcome:
			m_options=self.method_options_binary
		else:
			m_options=self.method_options
		tree_options={"criterion":'mse', "splitter":'best', "max_depth":None, "min_samples_split":2, "min_samples_leaf":1,
			"min_weight_fraction_leaf":0.0, "max_features":None, "random_state":None, "max_leaf_nodes":None, 
			"min_impurity_split":1e-07, "presort":False, "n_jobs": 1, "cv": 10, "search_range_low": 1, "search_range_high":11}
		if m_options!=None:
			if isinstance(m_options,dict):
				for i in m_options:
					if i in tree_options:
						tree_options[i]=m_options[i]
					else:
						print("Error: ",i," is not a valid option entry")
						print("valid option entries include:")
						for i in tree_options:
							print(i)
						raise NameError("Invalid option entry")	
						raise NameError("Invalid option entry")		
			else:
				raise NameError("options must take the form of a dictionary")
		#print (tree_options)
		criterion=tree_options["criterion"]
		splitter=tree_options["splitter"]
		min_samples_split=tree_options["min_samples_split"]
		min_weight_fraction_leaf=tree_options["min_weight_fraction_leaf"]
		max_features=tree_options["max_features"]
		random_state=tree_options["random_state"]
		max_leaf_nodes=tree_options["max_leaf_nodes"]
		min_impurity_split=tree_options["min_impurity_split"]
		presort=tree_options["presort"]
		n_jobs=tree_options["n_jobs"]
		cv=tree_options["cv"]
		search_range=range(tree_options["search_range_low"],tree_options["search_range_high"])
		parameters = {'max_depth':search_range}
		clf = sklearn.model_selection.GridSearchCV(tree.DecisionTreeRegressor(criterion=criterion, splitter=splitter,
				min_samples_split=min_samples_split, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features,
				random_state=random_state,max_leaf_nodes=max_leaf_nodes, presort=presort),
				parameters, n_jobs=n_jobs,cv=cv)
		return clf
	def _define_regression_tree_boosted(self,binary_outcome):
		if binary_outcome:
			m_options=self.method_options_binary
		else:
			m_options=self.method_options
		ada_options={"base_estimator":None, "n_estimators":100, "learning_rate":0.001, "loss":'exponential', "random_state":None}
		if m_options!=None:
			if isinstance(m_options,dict):
				for i in m_options:
					if i in ada_options:
						ada_options[i]=m_options[i]
					else:
						print("Error: ",i," is not a valid option entry")
						print("valid option entries include:")
						for i in ada_options:
							print(i)
						raise NameError("Invalid option entry")	
						raise NameError("Invalid option entry")		
			else:
				raise NameError("options must take the form of a dictionary")
		#print (ada_options)
		base_estimator=ada_options["base_estimator"]
		n_estimators=ada_options["n_estimators"]
		learning_rate=ada_options["learning_rate"]
		loss=ada_options["loss"]
		random_state=ada_options["random_state"]
		return sklearn.ensemble.AdaBoostRegressor(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate,
			loss=loss, random_state=random_state)
	def _define_ridge(self,binary_outcome):
		if binary_outcome:
			m_options=self.method_options_binary
		else:
			m_options=self.method_options
		ridge_options={"alphas":40, "fit_intercept":True, "scoring":None,
			"cv":10, "gcv_mode":None, "store_cv_values":False}
		if m_options!=None:
			if isinstance(m_options,dict):
				for i in m_options:
					if i in ridge_options:
						ridge_options[i]=m_options[i]
					else:
						print("Error: ",i," is not a valid option entry")
						print("valid option entries include:")
						for i in ridge_options:
							print(i)
						raise NameError("Invalid option entry")	
						raise NameError("Invalid option entry")		
			else:
				raise NameError("options must take the form of a dictionary")
		#print (ridge_options)
		alphas=ridge_options["alphas"]
		if (isinstance(alphas,int)):
			alphas=np.exp(np.linspace(-3*math.log(10),3*math.log(10),alphas))
		fit_intercept=ridge_options["fit_intercept"]
		scoring=ridge_options["scoring"]
		cv=ridge_options["cv"]
		gcv_mode=ridge_options["gcv_mode"]
		store_cv_values=ridge_options["store_cv_values"]
		return linear_model.RidgeCV(alphas=alphas, fit_intercept=fit_intercept,
				scoring=scoring, cv=cv, gcv_mode=gcv_mode, store_cv_values=store_cv_values)

	def _define_ridge_logit(self,binary_outcome):
		rl_options={"cv": 10, "Cs": 10, "solver":'SLSQP', "solver_options":None,"low_val": 1E-3, "high_val":1E3}
		m_options=self.method_options_binary
		if m_options!=None:
			if isinstance(m_options,dict):
				for i in m_options:
					if i in rl_options:
						rl_options[i]=m_options[i]
					else:
						print("Error: ",i," is not a valid option entry")
						print("valid option entries include:")
						for i in rl_options:
							print(i)
						raise NameError("Invalid option entry")	
						raise NameError("Invalid option entry")		
			else:
				raise NameError("options must take the form of a dictionary")
		#print (rl_options)
		cv=rl_options["cv"]
		Cs=rl_options["Cs"]
		solver=rl_options["solver"]
		solver_options=rl_options["solver_options"]
		low_val=rl_options["low_val"]
		high_val=rl_options["high_val"]
		return RidgeLogitCV(cv=cv, Cs=Cs, solver=solver, low_val=low_val, high_val=high_val)
	def _define_model(self,binary_outcome=False):
		if binary_outcome:
			local_method=self.method_binary
		else:
			local_method=self.method
		if local_method=="Tree":
			return self._define_regression_tree(binary_outcome)
		if local_method=="Random Forest":
			return self._define_random_forest(binary_outcome)
		if local_method=="Boosted Tree":
			return self._define_regression_tree_boosted(binary_outcome)
		if local_method=="Ridge":
			return self._define_ridge(binary_outcome)
		if local_method=="Ridge Logit":
			if binary_outcome:
				return self._define_ridge_logit(binary_outcome)
			else:
				raise NameError("The method 'Ridge Logit' can only be used to predict the binary outcome" )
		if local_method=="Lasso":
			return self._define_lasso(binary_outcome)
		if local_method=="Lasso Logit":
			if binary_outcome:
				return self._define_lasso_logit(binary_outcome)
			else:
				raise NameError("The method 'Lasso Logit' can only be used to predict the binary outcome" )
		else:
			raise NameError("You have entered an unrecognized machine learning mehod.\nRecognized methods include: `Lasso', `Ridge', `Tree' `Random Forest', and `Boosted Tree'")	
	
	def fit(self,X,Y,binary_outcome):
		"""
		The 'fit' method returns a fitted model of the regressors X on the outcome Y. If binary_outcome is True, then
		the fitted model corresponds to self.method_class_binary, with self.method_options_binary. If binary_outcome is False, then
		the fitted model corresponds to self.method_class, with self.method_options.
		Parameters
		----------
		X is a mxn numpy array where m is the number of observations and n is the number of regressors.
		Y is a row vector of length m.
		"""
		if binary_outcome:
			if self.method_binary=="Tree":
				return self.method_class_binary.fit(X=X,y=Y).best_estimator_
			else:
				return self.method_class_binary.fit(X=X,y=Y)
		else:
			if self.method_binary=="Tree":
				return self.method_class.fit(X=X,y=Y).best_estimator_
			else:
				return self.method_class.fit(X=X,y=Y)
	def _find_residuals(self, y_use,y_out,x_use,x_out,binary_outcome=False):
		"""
		The 'find_residuals' method uses the appropriate ml method (based on binary_outcomes value), and fits
		the ml method using the regressors x_use nad the outcome y_use. It then uses this same model to predict
		Parameters
		----------
		y_out given x_out, and returns the predicted values of y_out and the residuals of the predicted values
		x_use is an mxn numpy array where m is the number of observations and n is the number of regressors.
		y_use is a row vector of length m. The same idea applies to x_out and y_out.
		"""
		model=self.fit(x_use,y_use,binary_outcome=binary_outcome)
		yhat_out=model.predict(x_out)
		res_out=y_out-yhat_out
		return yhat_out, res_out
	def pl_estimate(self,X,y,d,test_size=.5, normalize=True,second_order_terms=False, verbose=True, standard_errors="Mackinnon"):
		"""
		The pl_estimate method is an implementation of the partial linear estimation procedure developed 
		in `Double Machine  Learning for Treatment and Causal Parameters' by Victor Chernozhukov,
		Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, and Whitney Newey. It is used to estimate
		the effect of the binary variable d on the outcome variable y, when X may be correlated with both d and y.

		Parameters
		----------
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
		verbose: boolean, optional (default=True).
			If set to true, then the beta and standard error results will be printed 
		standard_errors: string, optional (default="White")
			Options:
				-"Normal": results in normal standard errors
				-"White": results in heteroskedasticity robust standard errors
					as in White 1980
				-"Mackinnon": results in alternative heteroskedasticity robust standard errors
					as in Mackinnnon and White 1985
		"""
		start=time.time()
		if np.sum(abs(d**2-d))>0:
			raise NameError("The row vector 'd' can only have values of 0 or 1")
		if min(len(X),len(y),len(d))<max(len(X),len(y),len(d)):
			raise NameError("X, y,and d all must have the same length")
		if second_order_terms:
			X=self._so_terms(X)
		if normalize:
			X=self._normalize_matrix(X)
		y_col=np.array([y]).T
		d_col=np.array([d]).T
		data=np.concatenate((y_col,d_col,X),axis=1)
		split=sklearn.model_selection.train_test_split(data,test_size=test_size)
		data_use=split[0]
		y_use=data_use[:,0]
		d_use=data_use[:,1]
		x_use=data_use[:,2:]
		data_out=split[1]
		y_out=data_out[:,0]
		d_out=data_out[:,1]
		x_out=data_out[:,2:]
		exp_d_x,res_d_x=self._find_residuals(d_use,d_out,x_use,x_out,binary_outcome=True)
		exp_y_x,res_y_x=self._find_residuals(y_use,y_out,x_use,x_out,binary_outcome=False)
		ols_model=sm.regression.linear_model.OLS(res_y_x,res_d_x)
		reg=ols_model.fit()
		res=sm.regression.linear_model.RegressionResults(ols_model, reg.params, normalized_cov_params=None, scale=1.0, cov_type='nonrobust', cov_kwds=None, use_t=None)
		self.pl_beta=reg.params[0]
		if standard_errors=="Mackinnon":
			self.pl_se=res.HC1_se[0]
		elif standard_errors=="Normal":
			self.pl_se=res.bse[0]
		elif standard_errors=="White":
			self.pl_se=res.HC0_se[0]
		else:
			raise NameError("You have entered an unrecognized standard errors parameter value.\nRecognized values include: 'Mackinnon', Normal', and 'White' ")	
		if verbose:
			print("Partial Linear Estimate Results")
			print("----------------------------")
			print("Continous outcome machine learning method:", self.method)
			print ("Binary outcome machine learning method:", self.method_binary)
			print("Beta=", self.pl_beta)
			print("SE=", self.pl_se)
			print("Completed in", time.time()-start, "seconds")
		return self
	def interactive_estimate(self,X,y,d,test_size=.5,normalize=True, second_order_terms=False, drop_zero_divide=False, modify_zero_divide=1E-3,verbose=True):
		"""
		The interactive_estimate method is an implementation of the interactive estimation procedure developed 
		in `Double Machine  Learning for Treatment and Causal Parameters' by Victor Chernozhukov,
		Denis Chetverikov, Mert Demirer, Esther Duflo, Christian Hansen, and Whitney Newey. It is used to estimate
		the effect of the binary variable d on the outcome variable y, when X may be correlated with both d and y.

		Parameters
		----------
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
			is False. Whenever there is d_out[i]=1 and dhat[i]=0, dhat[i] is set to the value of modify_zero_divide.
			Similarly, whenever d_out[i]=0 and dhat[i]=1, then dhat[i] is set to the value of modify_zero_divide.
		"""
		start=time.time()
		if np.sum(abs(d**2-d))>0:
			raise NameError("The row vector 'd' can only have values of 0 or 1")
		if min(len(X),len(y),len(d))<max(len(X),len(y),len(d)):
			raise NameError("X, y,and d all must have the same length")
		if second_order_terms:
			X=self._so_terms(X)
		if normalize:
			X=self._normalize_matrix(X)
		y_col=np.array([y]).T
		d_col=np.array([d]).T
		data=np.concatenate((y_col,d_col,X),axis=1)
		split=sklearn.model_selection.train_test_split(data,test_size=test_size)
		data_use=split[0]
		y_use=data_use[:,0]
		d_use=data_use[:,1]
		x_use=data_use[:,2:]
		index_use1=((d_use==1).T)
		index_use0=(0==index_use1)
		data_use1=data_use[index_use1]
		y_use1=y_use[index_use1]
		x_use1=x_use[index_use1]
		y_use0=y_use[index_use0]
		x_use0=x_use[index_use0]
		data_out=split[1]
		y_out=data_out[:,0]
		d_out=data_out[:,1]
		x_out=data_out[:,2:]
		yhat_d1,resy_d1=self._find_residuals(y_use1,y_out,x_use1,x_out,binary_outcome=False)
		yhat_d0,resy_d0=self._find_residuals(y_use0,y_out,x_use0,x_out,binary_outcome=False)
		dhat,resd=self._find_residuals(d_use,d_out,x_use,x_out,binary_outcome=True)
		phi=yhat_d1-yhat_d0+d_out*resy_d1/dhat-((1-d_out)*resy_d0/(1-dhat))
		dhat_zero=np.nonzero(1*(dhat==0))[0]
		dhat_one=np.nonzero(1*(dhat==1))[0]
		if drop_zero_divide:
			bad_val_list=[]
			for i in dhat_zero:
				if d_out[i]==1:
					bad_val_list.append(i)	
				phi[i]=yhat_d1[i]-yhat_d0[i]-((1-d_out[i])*resy_d0[i]/(1-dhat[i]))
			for i in dhat_one:
				if d_out[i]==0:
					bad_val_list.append(i)
				phi[i]=yhat_d1[i]-yhat_d0[i]+d_out[i]*resy_d1[i]/dhat[i]
			phi=np.delete(phi,bad_val_list)
			self.interactive_beta=np.mean(phi)
			self.interactive_se=np.std(phi)/math.sqrt(len(phi))
			if verbose:
				print("Interactive Estimate Results")
				print("----------------------------")
				print("Continous outcome machine learning method:", self.method)
				print ("Binary outcome machine learning method:", self.method_binary)
				print("Beta=", self.interactive_beta)
				print("SE=", self.interactive_se)
				print("Completed in", time.time()-start, "seconds")
			return self
		else:
			for i in dhat_zero:
				if d_out[i]==1:
					d_out[i]=1-modify_zero_divide
				phi[i]=yhat_d1[i]-yhat_d0[i]-((1-d_out[i])*resy_d0[i]/(1-dhat[i]))
			for i in dhat_one:
				if d_out[i]==0:
					d_out[i]=modify_zero_divide
				phi[i]=yhat_d1[i]-yhat_d0[i]+d_out[i]*resy_d1[i]/dhat[i]
			self.interactive_beta=np.mean(phi)
			self.interactive_se=np.std(phi)/math.sqrt(len(phi))
			if verbose:
				print("Interactive Estimate Results")
				print("----------------------------")
				print("Continous outcome machine learning method:", self.method)
				print ("Binary outcome machine learning method:", self.method_binary)
				print("Beta=", self.interactive_beta)
				print("SE=", self.interactive_se)
				print("Completed in", time.time()-start, "seconds")
			return self
	def _so_terms(self,X):
		"""
		The function 'so_terms' creates a matrix featuring the first and second order terms for
		the input matrix X, where each row in X is an observation
		"""
		X_copy=np.copy(X)
		dim1=len(X_copy[0])
		total_count=int((dim1**2)/2+dim1/2)
		x_squared=np.zeros((len(X),total_count))
		zero_list=[]
		for i in range(dim1):
			for j in range(i,dim1):
				dim2=dim1-i
				index=total_count-int((dim2**2)/2+dim2/2)+j-i
				x_squared[:,index]=X_copy[:,i]*X_copy[:,j]
				if i==j:
					if np.sum(X_copy[:,i]-X_copy[:,i]**2)==0:
						zero_list.append(index)
		x_squared_mod=np.delete(x_squared,zero_list,1)
		return np.concatenate((X_copy,x_squared_mod),axis=1)
	def _normalize_matrix(self,X):
		"""
		The function 'normalize_matrix()' normalizes the input Matrix X by dividing each column in X
		by its standard deviation. X is an mxn numpy array where rows correspond to observations and 
		columns correspond to regressors
		"""
		X_copy=np.copy(X)
		for i in range(len(X_copy[0])):
			X_copy[:,i:i+1]=X_copy[:,i:i+1]/np.std(X_copy[:,i:i+1])
		return X_copy
from DoubleML import ML2Estimator
import pandas as pd
import numpy as np
import scipy.integrate as integrate

data = pd.read_stata('sipp1991.dta').values
data=np.delete(data,[0,2,11],1)
y_var=data[:,0]
d_var=data[:,8]
x_var=np.delete(data,[0,8],1)
method_list=np.array(["Lasso", "Ridge", "Random Forest", "Boosted Tree", "Tree"])

for i in range(5):
	if i<2:
		ML2Estimator(method=method_list[i]).pl_estimate(x_var,y_var,d_var,second_order_terms=True,verbose=True)
		print("")
		ML2Estimator(method=method_list[i]).interactive_estimate(x_var,y_var,d_var,second_order_terms=True,verbose=True)
		print("")
	else:
		ML2Estimator(method=method_list[i]).pl_estimate(x_var,y_var,d_var,second_order_terms=False,verbose=True)
		print("")
		ML2Estimator(method=method_list[i]).interactive_estimate(x_var,y_var,d_var,second_order_terms=False,verbose=True)
		print("")

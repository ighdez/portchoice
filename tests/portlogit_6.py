from random import seed
import pandas as pd
import numpy as np
from portchoice.models import PortLogit

# Load files
inputfile = 'data/toy_data_6.csv'
data = pd.read_csv(inputfile,index_col=None)

# Create arrays
Xvars = ['X_1','X_2','X_3']
J = 4
K = 3

Y = data[['Choice_' + str(i+1) for i in range(J)]]
C = None
X = data[[x + '_' + str(j+1) for j in range(J) for x in Xvars]]
B = None
Z = None

# Mutually-exclusive alternatives
ex = [np.array([1,2])]

# Create model
obj = PortLogit(Y,X,Z,C,B,ex)

# Estimate
asc = np.zeros(J).astype(int)
startv = np.ones(asc.sum() + K + 0)
fval, coef, se, hessian, diff_time = obj.estimate(startv,delta_0=0.,asc=asc,tol=1e-6,verbose=-1)
print(coef)

# Construct results matrix
results = pd.DataFrame(np.c_[coef,se,coef/se],columns=['Estimate','Std.Err.','T-stat'],index=Xvars)
print('\nEstimation results\nLog-likelihood: ' + str(round(-fval,2)) + '\n')
print(results)
print('\n')
# exit()
# Get the optimal portfolio
portfolio = obj.optimal_portfolio(X = X.mean(),Z = None, C = None, B = None, sims=10000)
print('Optimal portfolio:\n')
print(portfolio)
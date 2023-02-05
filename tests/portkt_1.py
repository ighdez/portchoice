from random import seed
import pandas as pd
import numpy as np
from portchoice.models import PortKT

# Load files
inputfile = 'data/toy_data_5.csv'
data = pd.read_csv(inputfile,index_col=None)

# Create arrays
Xvars = ['X_1','X_2','X_3']
J = 4
K = 3

Y = data[['Choice_' + str(i+1) for i in range(J)]]
C = data[['Cost_' + str(i+1) for i in range(J)]]
X = data[[x + '_' + str(j+1) for j in range(J) for x in Xvars]]
B = data['Budget'].iloc[0]

# Create model
obj = PortKT(Y,C,B,X)

# Estimate
asc = np.array([0,1,1,1])
startv = np.zeros(asc.sum() + K + 1)
fval, coef, se, hessian, diff_time = obj.estimate(startv,delta_0=None,sigma=1.,alpha_0=1.,gamma_0=1.,asc=asc,tol=1e-6,verbose=True)

# Construct results matrix
results = pd.DataFrame(np.c_[coef,se,coef/se],columns=['Estimate','Std.Err.','T-stat'],index=['ASC' + str(j+1) for j in range(asc.sum())] + Xvars + ['Cost'])
print('Log-likelihood: ' + str(round(-fval,1)) + '\n')
print(results)
exit()
# Get the optimal portfolio
portfolio = obj.optimal_portfolio(coef, C,B,X.mean().to_numpy(),None,asc=asc,delta_0=0,beta_j=None,sims=10000)

print(portfolio)
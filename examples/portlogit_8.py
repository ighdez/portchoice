from random import seed
import pandas as pd
import numpy as np
from portchoice.models import PortLogit

# Load files
inputfile = 'data/toy_data_8.csv'
data = pd.read_csv(inputfile,index_col=None)

# Create arrays
Xvars = ['X_1','X_2','X_3']
Zvars = ['Z_1','Z_2','Z_3']
J = 4
K = 3
M = 3

Y = data[['Choice_' + str(i+1) for i in range(J)]]
X = data[[x + '_' + str(j+1) for j in range(J) for x in Xvars]]
Z = data[Zvars]
C = None
B = None

# Create model
obj = PortLogit(Y,X,Z,C,B)

# Estimate
asc = np.ones(4).astype(int)
startv = np.zeros(asc.sum() + K + 0 + M*J)
fval, coef, se, hessian, diff_time = obj.estimate(startv,delta_0=0.,asc=asc,tol=1e-6,verbose=True,hess=False,method='bfgsmin')

# Construct results matrix
results = pd.DataFrame(np.c_[coef,se,coef/se],columns=['Estimate','Std.Err.','T-stat'],index=['ASC' + str(j+1) for j in range(asc.sum())] + Xvars + [z + '_' + str(j+1) for j in range(J) for z in Zvars])
print(results)
print(diff_time)
exit()
# Get the optimal portfolio
portfolio = obj.optimal_portfolio(coef, C,B,X.mean().to_numpy(),None,asc=asc,delta_0=0,beta_j=None,sims=10000)

print(portfolio)
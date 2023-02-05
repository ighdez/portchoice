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
C = None
X = data[[x + '_' + str(j+1) for j in range(J) for x in Xvars]]
B = None
Z = data[['Z_' + str(m+1) for m in range(M)]]

# Create model
obj = PortLogit(Y,X,Z,C,B)

# Estimate
asc = np.ones(4).astype(int)
startv = np.zeros(asc.sum() + K + M*J)
fval, coef, se, hessian, diff_time = obj.estimate(startv,delta_0=0.,asc=asc,hess=True,verbose=1)


# Construct results matrix
results = pd.DataFrame(np.c_[coef,se,coef/se],columns=['Estimate','Std.Err.','T-stat'],index=['ASC' + str(j+1) for j in range(J)] + Xvars + [z + '_a' + str(j+1) for j in range(J) for z in Zvars])
print('\nEstimation results\nLog-likelihood: ' + str(round(-fval,2)) + '\n')
print(results)
print('\n')
exit()
# Get the optimal portfolio
portfolio = obj.optimal_portfolio(X = X.mean(),Z = None, C = None, B = None, sims=10000)
print('Optimal portfolio:\n')
# print(portfolio)

print(coef)
print(hessian)
print(se)
print(coef/se)
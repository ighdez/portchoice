import pandas as pd
import numpy as np
from portchoice.generate import PortGen

# Set number of obs, alternatives, and atts
N = 10000
J = 4
K = 3
M = 3

# Set seed
np.random.seed(666)

# Set explanatory variables, costs and budget
X = np.random.uniform(size=(N,J,K))
Z = np.random.normal(size=(N,M))
C = None
B = None

# Set parameters
beta = np.array([-6,-3,10])
delta_j = np.array([1,2,3,4])
delta_0 = 0.
theta = np.array(
    [[0.1,0.2,0.3],
    [0.4,0.5,-0.6],
    [-0.1,0.2,-0.3],
    [0.4,0.5,-0.5]]
)

# Create utility
V = delta_j + X @ beta + Z @ theta.T

# Create PortGen obj
generator = PortGen(V,C,B,delta_0)

# Generate choices and log-likelihood
y, ll = generator.get_choices()

print('Log-likelihood of the full sample is ' + str(round(np.sum(ll),2)))

# Generate arrays
names = ['Choice_' + str(i+1) for i in range(J)] + ['X_' + str(k+1) + '_' + str(j+1) for j in range(J) for k in range(K)] + ['Z_' + str(m+1) for m in range(M)]
X_to_export = X.reshape((N,J*K))
to_export = np.c_[y,X_to_export,Z]
to_export = pd.DataFrame(to_export,columns=names)
to_export.to_csv('data/toy_data_8.csv')
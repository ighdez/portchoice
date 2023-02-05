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
C = None
B = None
Z = np.random.normal(size=(N,M))

# Set parameters
beta = np.array([-6,-3,10])
delta_0 = 0.
delta_j = np.array([1,2,3,4])
theta = np.c_[
    np.array([1,2,-3]),
    np.array([-1,-2,3]),
    np.array([2,4,-6]),
    np.array([-2,-4,6])]

# Create utility
V = X @ beta + delta_j + Z @ theta

# Create PortGen obj
generator = PortGen(V,C,B,delta_0)

# Generate choices
y, ll = generator.get_choices()

# Compute log-likelihood
print('Log-likelihood of the full sample is ' + str(round(np.sum(ll),2)))

# Generate arrays
names = ['Choice_' + str(i+1) for i in range(J)] + ['X_' + str(k+1) + '_' + str(j+1) for j in range(J) for k in range(K)] + ['Z_' + str(m+1) for m in range(M)]
X_to_export = X.reshape((N,J*K))
to_export = np.c_[y,X_to_export,Z]
to_export = pd.DataFrame(to_export,columns=names)
to_export.to_csv('data/toy_data_8.csv')
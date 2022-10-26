import pandas as pd
import numpy as np
from portchoice.generate import PortGen

# Set number of obs, alternatives, and atts
N = 10000
J = 4
K = 3

# Set seed
np.random.seed(666)

# Set explanatory variables, costs and budget
X = np.random.uniform(size=(N,J,K))
C = None
B = None

# Set parameters
beta = np.array([-6,-3,10])
delta_0 = 0.

# Create utility
V = X @ beta

# Mutually-exclusive alternatives
ex = [np.array([1,2])]

# Create PortGen obj
generator = PortGen(V,C,B,delta_0,ex)

# Generate choices and log-likelihood
y, ll = generator.get_choices()

print('Log-likelihood of the full sample is ' + str(round(np.sum(ll),2)))

# Generate arrays
names = ['Choice_' + str(i+1) for i in range(J)] + ['X_' + str(k+1) + '_' + str(j+1) for j in range(J) for k in range(K)]# + ['Cost_' + str(i+1) for i in range(J)] + ['Budget'] 
X_to_export = X.reshape((N,J*K))
to_export = np.c_[y,X_to_export]
to_export = pd.DataFrame(to_export,columns=names)
to_export.to_csv('data/toy_data_6.csv')
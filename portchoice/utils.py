"""BFGSMin functions"""
# BFGSMin functions
# Written by Jose Ignacio Hernandez

# Import packages
import numpy as np

# Numeric gradient fuction
def _numgr(f,param,difftype='forward',eps=np.sqrt(np.finfo(float).eps),*args):    
    
    # Define scalars and initialize vectors
    K = len(param)                          # No. of parameters
    gr = np.full(K,np.nan)                  # Initialize gradient vector
    ej = np.eye(K)*eps                      # Vector of eps
    
    # If difftype == 'central', then:
    if difftype == 'central':
        for k in range(K):
            gr[k] = (f(param + ej[:,k],*args) - f(param - ej[:,k],*args))*0.5/eps

    # ...else, if difftype == 'forward' (default):
    elif difftype == 'forward':
        f0 = f(param,*args)
        for k in range(K):
            gr[k] = (f(param + ej[:,k],*args) - f0)/eps

    # ...else, return error
    else:
        raise ValueError('''difftype must be either 'forward' or 'central' ''')

    # Return gradient vector
    return(gr)

# BFGS Minimizer function
def _bfgsmin(f,x0,maxiter=1000,tol=np.sqrt(np.finfo(float).eps),verbose=False,difftype='central',diffeps=np.sqrt(np.finfo(float).eps),steptol=1e-30,args=()):

    # Ignore warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Initialize parameters
    x = np.array(x0)                            # Initial value of x
    f_val = f(x,*args)                          # Initial value for objective function
    g0 = _numgr(f,x,difftype,diffeps,*args)      # Initial value of gradient
    H0 = np.eye(x.shape[0])                     # Initial value of approximate Hessian is a positive definite matrix (e.g. identity matrix)
    g_diff = np.inf                             # Initial difference in gradient norm is set to infinity
    c1 = 1e-4                                   # Internal scalar for the Armijo-Goldstein condition (for step size computation)
    lambd = 1                                   # Initial value of the step size
    convergence = -1                            # Set convergence flag to -1. If zero, the algorithm converged

    # Print inital value of the objective function
    if verbose:
        print('Initial F-value: ' + str(round(f_val,2)))

    # Start algorithm
    for iter in range(maxiter):
        
        # If gradient norm is less than tolerance value, convergence is achieved and the loop is broken
        if g_diff < tol:
            convergence = 0

            if verbose:
                print('\nLocal minimum found. G-norm below tolerance')

            break

        # Set step size = 1 on each new iteration
        lambd = 1

        # Construct direction vector and relative gradient

        # d = np.linalg.solve(-H0,np.eye(len(x))) @ g0
        d = np.linalg.inv(-H0) @ g0
        m = d.T @ g0
        # Select step size that satisfies the Armijo-Goldstein condition
        while True:
            
            # If lambd decreases less than tol, then stop and return convergence = 3
            if lambd < steptol:
                if verbose:
                    print('\nLocal minimum possible. Step size tolerance limit reached.')
                convergence = 5
                break

            # Evaluate objective function using current step size
            x1 = x + lambd*d
            f1 = f(x1,*args)

            # Construct test for A-G condition
            ftest = f_val + c1*lambd*m

            # If A-G condition is satisfied (i.e current step size generates a decrease of the objective function), continue with BFGS algorithm
            if (np.isnan(f1) == False) and (f1 <= ftest) and (f1 > 0):
                break

            # ...else, decrease step size
            else:
                lambd = lambd/2

        # If step size calculation loop was broken, then break the algorithm and return convergence code + partial results
        if convergence == 5:
            break

        # BFGS ALGORITHM: construct the improvement and gradient improvement
        x = x1
        g1 = _numgr(f,x,difftype,diffeps,*args)
        s0 = (lambd*d)[:,np.newaxis]
        y0 = (g1 - g0)[:,np.newaxis]

        # Update Hessian using BFGS formula
        H0 = H0 + (y0 @ y0.T) / (y0.T @ s0) - (((H0 @ s0) @ s0.T) @ H0)/((s0.T @ H0) @ s0)
        
        # Store new gradient and compute the new value of objective function
        g0 = g1
        f_val = f(x,*args)

        # Compute infinite norm of gradient vector 
        g_diff = np.abs(m)
        # Print output
        if verbose:
            print('Iter No. ' + str(int(iter+1)) + ': F-value: ' + str(round(f_val,2)) + ' / Step size: ' + str(round(lambd,6)) + ' / G-norm: ' + str(round(g_diff,6)))

    # If the algorithm reaches the maximum iterations, it prints convergence message = 2
    if iter == maxiter:
        convergence = 2

    # Return the Hessian approximation
    H = H0

    # Return convergence flag, iterations, final f value, final x value, and final approx. hessian
    return({'convergence': convergence, 'iterations': iter+1, 'fun': f_val, 'x': x, 'hessian': H})
"""Portfolio choice model functions."""
# Portfolio models
# Written by Jose Ignacio Hernandez
# May 2022

# Load required modules
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from numdifftools import Hessian
from pyDOE2 import fullfact
from portchoice.utils import _bfgsmin
import time

# Portfolio Logit model
class PortLogit:
    """Portfolio logit model class.

    It contains the routines to prepare the data and estimate a portfolio 
    logit model, as well as for the computation of the optimal portfolio.

    Parameters
    ----------
    Y : pd.DataFrame
        A data frame with choices of each alternative for each respondent.
    X : pd.DataFrame, optional
        A data frame with the alternative-specific variables 
        (e.g., attributes), by default None
    Z : pd.DataFrame, optional
        A data frame with the individual-specific variables, by default None
    C : pd.DataFrame, optional
        A data frame with the costs of each individual alternative for each 
        respondent, by default None
    B : float, optional
        Resource constraint, by default None
    mutually_exclusive : list, optional
        List of mutually-exclusive alternatives. Each element of the list is
        a numpy array of two elements that detail the two mutually-exclusive
        alternatives.
    """
    # Init function
    def __init__(self,Y: pd.DataFrame, X: pd.DataFrame = None, Z: pd.DataFrame = None, C: pd.DataFrame = None, B: float = None, mutually_exclusive: list = None):

        # Array of choices
        self.Y = Y.to_numpy()
        
        # Get scalars N and J
        self.N = self.Y.shape[0]
        self.J = self.Y.shape[1]

        # Calculate combinations array
        self.combinations = fullfact(np.repeat(2,self.J))

        # If mutually-exclusive alternatives are defined, then set utility to -inf
        self.mutually_exclusive = mutually_exclusive
        
        if mutually_exclusive is not None:
            
            idx = []

            # Loop across combinations of mutually-exclusive alts
            for e in mutually_exclusive:

                # Find indexes
                e_j = e - 1
                idx.append(np.where((self.combinations[:,e_j[0]]==1) & (self.combinations[:,e_j[1]]==1))[0])

            # Remove mutually-exclusive alternatives
            idx = np.unique(np.concatenate(idx))

            self.combinations = np.delete(self.combinations,idx,axis=0)

        # Define array for alternative-specific covariates and shape K (if present)
        if X is not None:
            self.K = int(X.shape[1]/self.J)
            self.X = X.to_numpy().reshape((self.N,self.J,self.K))
        else:
            self.K = 0
            self.X = None

        # Define array of individual-specific covariates (if present)
        if Z is not None:
            self.Z = Z.to_numpy()
        else:
            self.Z = None

        # Define array or budget scalar and feasible combinations (if present)
        self.B = B

        # Define arrays of costs and totalcosts (if present)
        if C is not None:
            self.C = C.to_numpy()
            self.Totalcosts = self.C @ self.combinations.T
            
            if B is not None:
                self.Feasible = (self.Totalcosts <= self.B)
            else:
                self.Feasible = np.ones(self.Totalcosts.shape)                
        else:
            self.C = 0
            self.Totalcosts = 0.
            self.Feasible = np.ones(self.combinations.shape)  

    # Estimate portfolio logit model
    def estimate(self, startv: np.ndarray, asc: np.ndarray, beta_j: np.ndarray = None, delta_0: float = None, method: str = 'bfgsmin', hess: bool = True, tol: float = 1e-6, verbose: bool = True):
        """Estimate portfolio logit model

        It starts the optimisation routine of the portfolio logit model. 
        The user can specify the presence of alternative-specific constants 
        (`asc`), separate parameters for the alternative-specific variables 
        (`beta_j`) and the presence of a parameter that captures the marginal 
        utility of non-spent resources (`delta_0`).

        Parameters
        ----------
        startv : np.ndarray
            Starting values for the maximum-likelihood estimation routine.
        asc : np.ndarray
            An array of length `n_alternatives`, in which each element can
            be either equal to one if the ASC of the corresponding alternative 
            is estimated, and zero otherwise.
        beta_j : np.ndarray, optional
            An array of dimension `n_alternatives*n_attributes`, in which each 
            element can be either equal to one if the corresponding 
            alternative-specific parameter is estimated, and zero otherwise. 
            If `beta_j = None` and `X` exists then single attribute-specific 
            parameters (i.e., equal across alternatives) are estimated 
            , by default None
        delta_0 : float, optional
            If None and `C` exists, then the parameter of the marginal utility 
            of non-spent resources is estimated. If `delta_0` is a float, then 
            the parameter is fixed to the value of `delta_0`, by default None
        method : str, optional
            The optimisation method for the MLE routine. Available options are 
            either the built-in BFGS minimiser ('bfgsmin') or a method available 
            for `scipy.minimize.optimize`, by detault 'bfgsmin'
        hess : bool, optional
            Whether the finite-difference hessian is estimated at the end of the 
            estimation routine, by default True
        tol : float, optional
            Tolerance of the gradient in the estimation routine, by default 1e-6
        verbose : bool, optional
            Whether the estimation routine returns information at each iteration. 
            See the documentation of `scipy.optimize.minimize` with method 
            `l-bfgs-b` for more information, by default True

        Returns
        -------
        ll : float
            Log-likelihood function at the optimum
        coef : numpy.ndarray
            Estimated parameters at the optimum
        se : numpy.ndarray
            Standard errors of `coef`. If `hess = False` then `se = 0.`
        hessian : numpy.ndarray
            Finite-difference Hessian. If `hess = False` then `hessian = 0.`
        diff_time : float
            Estimation time in seconds.
        """
        # Retrieve parameter specifications and store in object
        self.asc = asc
        self.beta_j = beta_j
        self.delta_0 = delta_0

        # Set arguments for the estimation routine
        args = (self.J,self.K,self.Y,self.C,self.B,self.X,self.Z,self.combinations,self.Totalcosts,self.Feasible,self.asc,self.delta_0,self.beta_j)
            
        # Minimise the LL function
        time0 = time.time()

        if method == 'bfgsmin':
            res = _bfgsmin(PortLogit._llf,startv,tol=tol,verbose=verbose,difftype='forward',args=args)
        else:
            res = minimize(PortLogit._llf,startv,args=args,method='L-BFGS-B',options={'gtol': tol, 'iprint': verbose})
        
        # Get/compute outputs
        ll = res['fun']
        self.coef = res['x'].flatten()

        if verbose > -1:
            print('Computing Hessian')

        if hess:
            hessian = Hessian(PortLogit._llf)(self.coef,self.J,self.K,self.Y,self.C,self.B,self.X,self.Z,self.combinations,self.Totalcosts,self.Feasible,asc,delta_0,beta_j)
            se = np.sqrt(np.diag(np.linalg.inv(hessian))).flatten()
        else:
            if method == 'bfgsmin':
                hessian = res['hessian']
                se = np.sqrt(np.diag(np.linalg.inv(hessian))).flatten()
            else:
                hessian = 0.
                se = 0.

        time1 = time.time()
        diff_time = time1-time0

        # Return results
        return ll, self.coef, se, hessian, diff_time

    # Optimal portfolio
    def optimal_portfolio(self,X: pd.Series = None, Z: pd.DataFrame = None, C: pd.Series = None, B: float = None,sims: int = 1000):
        """Compute the optimal portfolio

        Computes the optimal portfolio based on the estimation results 
        (i.e., obtained from `estimate()`) and user-defined variables. 
        The optimal portfolio is computed by computing the expected 
        utility of all possible combinations of alternatives. The 
        expected utility is computed by simulation using `sims` error 
        draws from an Extreme Value (Gumbel) distribution.

        Parameters
        ----------
        X : pd.Series, optional
            Series of alternative-specific variables, by default None
        Z : pd.DataFrame, optional
            Data frame with individual-specific variables, by default None
        C : pd.Series, optional
            Series with individual costs per alternative, by default None
        B : float, optional
            Resource constraint, by default None
        sims : int, optional
            Number of Extreme Value random draws, by default 1000

        Returns
        -------
        portfolio : pd.DataFrame
            Data frame with the optimal portfolio (ranked combinations),
            its expected utility and its total cost (if `C` is not None).
        """
        # Define array for alternative-specific covariates and shape K (if present)
        if X is not None:
            X = X.to_numpy().reshape((1,self.J,self.K))
        else:
            X = None

        # Define arrays of costs and totalcosts (if present)
        if C is not None:
            Totalcosts = (self.combinations * C.to_numpy()).sum(axis=1)[np.newaxis,:]
            if B is not None:
                Feasible = Totalcosts <= B
            else:
                Feasible = np.ones((1,self.combinations.shape[0])).astype(bool)
        else:
            Totalcosts = 0.
            Feasible = np.ones((1,self.combinations.shape[0]))

        # Create random Gumbel draws
        e = np.random.gumbel(size=(sims,self.combinations.shape[0]))

        # Get utility of each portfolio
        Vp = _utility(self.coef,self.J,self.K,None,C,B,X,Z,self.combinations,Totalcosts,Feasible,self.asc,self.delta_0,self.beta_j,return_chosen=False)

        # Compute utility for each simulation and average
        Up_s = Vp + e
        Up = Up_s.mean(axis=0)

        # Set utility of unfeasible combinations as -inf
        if B is not None:
            Up[~Feasible] = -np.inf

        # Sort portfolios and costs by expected utility
        sort_index = np.argsort(Up)[::-1]
        combinations_sorted = self.combinations[sort_index,:]
        EU_sorted = Up[sort_index]

        # If costs are present, add to the frame and drop unfeasible combinations
        if C is not None:
            Totalcosts_sorted = Totalcosts[sort_index]

            Totalcosts_sorted[EU_sorted != -np.inf]
            combinations_sorted = combinations_sorted[EU_sorted != -np.inf]
            EU_sorted = EU_sorted[EU_sorted != -np.inf]

        # Construct dataframe with expected utility
        portfolio_columns = ['Alt_' + str(i+1) for i in range(self.J)]
        portfolio = pd.concat([ pd.DataFrame(combinations_sorted,columns=portfolio_columns),
                                pd.Series(EU_sorted,name='EU')],axis=1)
        
        if C is not None:
            portfolio = pd.concat([portfolio,pd.Series(Totalcosts_sorted,name='Totalcosts')],axis=1)

        # Return pandas dataframe
        return portfolio

    # Portfolio choice model log-likelihood function
    @staticmethod
    def _llf(pars,J,K,Y,C,B,X,Z,combinations,Totalcosts,Feasible,asc,delta_0,beta_j):
                
        # Get utility functions of chosen alternatives and of portfolios
        Vp, Vp_chosen = _utility(pars,J,K,Y,C,B,X,Z,combinations,Totalcosts,Feasible,asc,delta_0,beta_j, return_chosen = True)

        # Clip to avoid numerical overflow
        Vp[Vp>700] = 700
        Vp_chosen[Vp_chosen>700] = 700
        
        prob_1 = np.exp(Vp_chosen)
        prob_2 = np.sum(np.exp(Vp),axis=1)

        # Get choice probability
        probs = prob_1/prob_2
        probs[~np.isfinite(probs)] = 1

        # Log-likelihood is the negative of the sum of LN of choice probabilities
        ll = -np.sum(np.log(probs))

        # Return log-likelihood
        return ll

# Portfolio Logit model
class LCPortLogit:
    """Latent class Portfolio logit model class.

    It contains the routines to prepare the data and estimate a latent 
    class portfolio logit model, as well as for the computation of the 
    optimal portfolio.

    Parameters
    ----------
    Y : pd.DataFrame
        A data frame with choices of each alternative for each respondent.    
    lc : int
        Number of latent classes.    
    X : pd.DataFrame, optional
        A data frame with the alternative-specific variables 
        (e.g., attributes), by default None
    Z : pd.DataFrame, optional
        A data frame with the individual-specific variables, by default None
    C : pd.DataFrame, optional
        A data frame with the costs of each individual alternative for each 
        respondent, by default None
    B : float, optional
        Resource constraint, by default None
    mutually_exclusive : list, optional
        List of mutually-exclusive alternatives. Each element of the list is
        a numpy array of two elements that detail the two mutually-exclusive
        alternatives.
    """
    # Init function
    def __init__(self,Y: pd.DataFrame, lc: int, X: pd.DataFrame = None, Z: pd.DataFrame = None, C: pd.DataFrame = None, B: float = None, mutually_exclusive: list = None):

        # Array of choices
        self.Y = Y.to_numpy()

        # Latent classes
        self.lc = lc

        # Get scalars N and J
        self.N = self.Y.shape[0]
        self.J = self.Y.shape[1]

        # Calculate combinations array
        self.combinations = fullfact(np.repeat(2,self.J))

        # If mutually-exclusive alternatives are defined, then set utility to -inf
        self.mutually_exclusive = mutually_exclusive
        
        if mutually_exclusive is not None:
            
            idx = []

            # Loop across combinations of mutually-exclusive alts
            for e in mutually_exclusive:

                # Find indexes
                e_j = e - 1
                idx.append(np.where((self.combinations[:,e_j[0]]==1) & (self.combinations[:,e_j[1]]==1))[0])

            # Remove mutually-exclusive alternatives
            idx = np.unique(np.concatenate(idx))

            self.combinations = np.delete(self.combinations,idx,axis=0)

        # Define array for alternative-specific covariates and shape K (if present)
        if X is not None:
            self.K = int(X.shape[1]/self.J)
            self.X = X.to_numpy().reshape((self.N,self.J,self.K))
        else:
            self.K = 0
            self.X = None

        # Define array of individual-specific covariates (if present)
        if Z is not None:
            self.Z = Z.to_numpy()
        else:
            self.Z = None

        # Define array or budget scalar and feasible combinations (if present)
        self.B = B

        # Define arrays of costs and totalcosts (if present)
        if C is not None:
            self.C = C.to_numpy()
            self.Totalcosts = self.C @ self.combinations.T
            
            if B is not None:
                self.Feasible = (self.Totalcosts <= self.B)
            else:
                self.Feasible = np.ones(self.Totalcosts.shape)                
        else:
            self.C = 0
            self.Totalcosts = 0.
            self.Feasible = np.ones(self.combinations.shape)  

    # Estimate latent class portfolio logit model
    def estimate(self, startv: np.ndarray, asc: np.ndarray, beta_j: np.ndarray = None, delta_0: float = None, hess: bool = True, tol: float = 1e-6, verbose: bool = True):
        """Estimate latent class portfolio logit model

        It starts the optimisation routine of the latent class portfolio logit model. 
        The user can specify the presence of alternative-specific constants 
        (`asc`), separate parameters for the alternative-specific variables 
        (`beta_j`) and the presence of a parameter that captures the marginal 
        utility of non-spent resources (`delta_0`).

        Parameters
        ----------
        startv : np.ndarray
            Starting values for the maximum-likelihood estimation routine.
        asc : np.ndarray
            An array of length `n_alternatives`, in which each element can
            be either equal to one if the ASC of the corresponding alternative 
            is estimated, and zero otherwise.
        beta_j : np.ndarray, optional
            An array of dimension `n_alternatives*n_attributes`, in which each 
            element can be either equal to one if the corresponding 
            alternative-specific parameter is estimated, and zero otherwise. 
            If `beta_j = None` and `X` exists then single attribute-specific 
            parameters (i.e., equal across alternatives) are estimated 
            , by default None
        delta_0 : float, optional
            If None and `C` exists, then the parameter of the marginal utility 
            of non-spent resources is estimated. If `delta_0` is a float, then 
            the parameter is fixed to the value of `delta_0`, by default None
        hess : bool, optional
            Whether the finite-difference hessian is estimated at the end of the 
            estimation routine, by default True
        tol : float, optional
            Tolerance of the gradient in the estimation routine, by default 1e-6
        verbose : bool, optional
            Whether the estimation routine returns information at each iteration. 
            See the documentation of `scipy.optimize.minimize` with method 
            `l-bfgs-b` for more information, by default True

        Returns
        -------
        ll : float
            Log-likelihood function at the optimum
        coef : numpy.ndarray
            Estimated parameters at the optimum
        se : numpy.ndarray
            Standard errors of `coef`. If `hess = False` then `se = 0.`
        hessian : numpy.ndarray
            Finite-difference Hessian. If `hess = False` then `hessian = 0.`
        diff_time : float
            Estimation time in seconds.
        """
        # Retrieve parameter specifications and store in object
        self.asc = asc
        self.beta_j = beta_j
        self.delta_0 = delta_0

        # Set arguments for the estimation routine
        args = (self.J,self.K,self.Y,self.C,self.B,self.X,self.Z,self.combinations,self.Totalcosts,self.Feasible,self.asc,self.delta_0,self.beta_j,self.lc)
            
        # Minimise the LL function using BFGSmin
        time0 = time.time()
        res = minimize(LCPortLogit._llf,startv,args=args,method='L-BFGS-B',options={'gtol': tol, 'iprint': verbose})
        
        # Get/compute outputs
        ll = res['fun']
        self.coef = res['x'].flatten()

        if verbose > -1:
            print('Computing Hessian')

        if hess:
            hessian = Hessian(LCPortLogit._llf)(self.coef,self.J,self.K,self.Y,self.C,self.B,self.X,self.Z,self.combinations,self.Totalcosts,self.Feasible,self.asc,self.delta_0,self.beta_j,self.lc)
            se = np.sqrt(np.diag(np.linalg.inv(hessian))).flatten()
        else:
            hessian = 0.
            se = 0.

        time1 = time.time()
        diff_time = time1-time0

        # Return results
        return ll, self.coef, se, hessian, diff_time

    # Optimal portfolio
    def optimal_portfolio(self,X: pd.Series = None, Z: pd.DataFrame = None, C: pd.Series = None, B: float = None,sims: int = 1000):
        """Compute the optimal portfolio

        Computes the optimal portfolio based on the estimation results 
        (i.e., obtained from `estimate()`) and user-defined variables. 
        The optimal portfolio is computed by computing the expected 
        utility of all possible combinations of alternatives. The 
        expected utility is computed by simulation using `sims` error 
        draws from an Extreme Value (Gumbel) distribution.

        Parameters
        ----------
        X : pd.Series, optional
            Series of alternative-specific variables, by default None
        Z : pd.DataFrame, optional
            Data frame with individual-specific variables, by default None
        C : pd.Series, optional
            Series with individual costs per alternative, by default None
        B : float, optional
            Resource constraint, by default None
        sims : int, optional
            Number of Extreme Value random draws, by default 1000

        Returns
        -------
        portfolio : pd.DataFrame
            Data frame with the optimal portfolio (ranked combinations),
            its expected utility and its total cost (if `C` is not None).
        """
        # Define array for alternative-specific covariates and shape K (if present)
        if X is not None:
            X = X.to_numpy().reshape((1,self.J,self.K))
        else:
            X = None

        # Define arrays of costs and totalcosts (if present)
        if C is not None:
            Totalcosts = (self.combinations * C.to_numpy()).sum(axis=1)[np.newaxis,:]
            if B is not None:
                Feasible = Totalcosts <= B
            else:
                Feasible = np.ones((1,self.combinations.shape[0])).astype(bool)
        else:
            Totalcosts = 0.
            Feasible = np.ones((1,self.combinations.shape[0]))

        # Create random Gumbel draws
        e = np.random.gumbel(size=(sims,self.combinations.shape[0]))

        # Get utility of each portfolio
        Vp = _utility(self.coef,self.J,self.K,None,C,B,X,Z,self.combinations,Totalcosts,Feasible,self.asc,self.delta_0,self.beta_j,return_chosen=False)

        # Compute utility for each simulation and average
        Up_s = Vp + e
        Up = Up_s.mean(axis=0)

        # Set utility of unfeasible combinations as -inf
        if B is not None:
            Up[~Feasible] = -np.inf

        # Sort portfolios and costs by expected utility
        sort_index = np.argsort(Up)[::-1]
        combinations_sorted = self.combinations[sort_index,:]
        EU_sorted = Up[sort_index]

        # If costs are present, add to the frame and drop unfeasible combinations
        if C is not None:
            Totalcosts_sorted = Totalcosts[sort_index]

            Totalcosts_sorted[EU_sorted != -np.inf]
            combinations_sorted = combinations_sorted[EU_sorted != -np.inf]
            EU_sorted = EU_sorted[EU_sorted != -np.inf]

        # Construct dataframe with expected utility
        portfolio_columns = ['Alt_' + str(i+1) for i in range(self.J)]
        portfolio = pd.concat([ pd.DataFrame(combinations_sorted,columns=portfolio_columns),
                                pd.Series(EU_sorted,name='EU')],axis=1)
        
        if C is not None:
            portfolio = pd.concat([portfolio,pd.Series(Totalcosts_sorted,name='Totalcosts')],axis=1)

        # Return pandas dataframe
        return portfolio

    # Log-likelihood function
    @staticmethod
    def _llf(pars,J,K,Y,C,B,X,Z,combinations,Totalcosts,Feasible,asc,delta_0,beta_j,lc):

        # Separate class parameters and transform to probabilities (last lc-1 pars are the class membership parameters. First lc par is fixed in zero)
        pars_classes = np.concatenate((0.,pars[-(lc-1):]),axis=None)
        probs_classes = np.exp(pars_classes)/np.sum(np.exp(pars_classes))

        # Set utility parameters in a list
        pars_utility = []
        lc_count = 0
        npars = int(len(pars[:-(lc-1)])/lc)
        for c in range(lc):
            pars_utility.append(pars[lc_count:(lc_count+npars)])
            lc_count += npars

        # Get utility of chosen alternatives and of portfolios per class
        probs_lc = []

        for c in range(lc):
            Vp, Vp_chosen = _utility(pars_utility[c],J,K,Y,C,B,X,Z,combinations,Totalcosts,Feasible,asc,delta_0,beta_j, return_chosen = True)

            # Clip to avoid numerical overflow
            Vp[Vp>700] = 700
            Vp_chosen[Vp_chosen>700] = 700
        
            prob_1 = np.exp(Vp_chosen)
            prob_2 = np.sum(np.exp(Vp),axis=1)

            # Get choice probability
            probs_c = prob_1/prob_2
            probs_c[~np.isfinite(probs_c)] = 1

            probs_lc.append(probs_c)
        
        probs_lc = np.vstack(probs_lc).T

        # Create unconditional probabilities
        probs = np.sum(probs_classes * probs_lc, axis=1)

        # Log-likelihood is the negative of the sum of LN of choice probabilities
        ll = -np.sum(np.log(probs))

        # Return log-likelihood
        return ll

# Utility functions method
def _utility(pars,J,K,Y,C,B,X,Z,combinations,Totalcosts,Feasible,asc,delta_0,beta_j,return_chosen=True):

            # Separate parameters of pars
            par_count = 0

            # Alternative-specific constants
            delta_j = np.zeros(J)
            for j in range(J):
                if asc[j] == 1:
                    delta_j[j] = pars[par_count]
                    par_count += 1

            # Attribute-specific parameters
            if X is not None:
                if beta_j is not None:
                    beta = np.zeros(beta_j.shape)
                    for j in range(J):
                        for k in range(K):
                            if beta_j[j,k] == 1:
                                beta[j,k] = pars[par_count]
                                par_count += 1
                    Xb = np.sum(X * beta,axis=2)
                else:
                    beta = pars[par_count:(K+par_count)]                
                    Xb = X @ beta
                    par_count += K
            else:
                beta= 0.
                Xb = 0.

            # Individual-specific parameters
            if Z is not None:
                theta = np.vstack([np.zeros(Z.shape[1]), pars[par_count:].reshape(((J-1),Z.shape[1]))])
                Zt = Z @ theta.T
            else:
                theta = 0.
                Zt = 0.

            # Cost parameter
            if delta_0 is None:
                delta_0 = pars[par_count]
                par_count += 1

            # Construct individual utility functions
            Vj = delta_j + Xb + Zt

            # Construct utility functions of the portfolios
            Vp = Vj @ combinations.T

            if return_chosen:
                Vp_chosen = np.sum(Vj*Y,axis=1)

            if B is not None:
                Vp += delta_0*(B-Totalcosts)
                Vp[~Feasible] = -np.inf
                if return_chosen:
                    Vp_chosen += delta_0*(B-np.sum(C*Y,axis=1))
            else:
                Vp -= delta_0*Totalcosts
                if return_chosen:
                    Vp_chosen -= delta_0*np.sum(C*Y,axis=1)
        
            # Return utility functions
            if return_chosen:
                return Vp, Vp_chosen
            else:
                return Vp
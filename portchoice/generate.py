"""Portfolio choice data generation functions."""
from itertools import combinations
import numpy as np
from pyDOE2 import fullfact

# Portfolio data generator class
class PortGen:
    """Portfolio choice data generator class.

    This class generates synthetic choices of a portfolio choice model. It 
    takes the utility of each individual alternative and generates synthetic 
    choices and the 'true' log-likelihood function. `PortGen` allows for 
    unconstrained and constrained choice situations (i.e., with resource 
    constraints).

    Parameters
    ----------
    V : np.ndarray
        Array of deterministic utilities of each individual alternative.
    C : np.ndarray, optional
        Array of costs of each individual alternative, by default None
    delta_0 : float, optional
        Parameter of the marginal utility of non-spent resources. Must be 
        different from None if `C` is defined, by default None
    B : float, optional
        Available resources. Must be different from None if `C` is defined, 
        by default None
    mutually_exclusive : list, optional
        List of mutually-exclusive alternatives. Each element of the list is
        a numpy array of two elements that detail the two mutually-exclusive
        alternatives.
    """
    # Init function
    def __init__(self, V: np.ndarray, C: np.ndarray = None, delta_0: float = None, B: float = None, mutually_exclusive: list = None):

        # Number of individual choices
        J = V.shape[1]

        # Create matrix of combinations
        self.combinations = fullfact(np.repeat(2,J))

        # If mutually-exclusive alternatives are defined, then set utility to -inf
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

        # Create utility of each feasible combination
        self.Vp = V @ self.combinations.T

        # If costs array is present, create total costs and (dis-)utilty of spending resources
        if C is not None:
            assert delta_0 is not None, "If costs array (C) is \
            present, then non-spent resources parameter (delta_0) must \
            be defined."

            # Create total costs and (dis-)utilty of spending resources
            Totalcosts = C @ self.combinations.T
            Vp_costs = -delta_0*Totalcosts
            
            # If a resource constraint is present, set unfeasible combinations as -inf
            if B is not None:
                Vp_costs += delta_0*B
                Vp_costs[Totalcosts>B] = -np.inf

            # Add (dis-)utilty of spending resources
            self.Vp += Vp_costs

    # Get synthetic choices and log-likelihood
    def get_choices(self):
        """Generate portfolio synthetic choices and log-likelihood.

        It takes the configurations parameters of `PortGen` and generates 
        a `numpy` array that contains the pseudo-synthetic choices for 
        each observation. Additionally, it returns the 'true' log-likelihood.

        Returns
        -------
        y : numpy.ndarray
            A `numpy` array with the synthetic choices for each observation.
        ll : float
            The 'true' log-likelihood.
        """
        # Compute EV1 errors
        e = np.random.gumbel(size=self.Vp.shape)

        # Compute utility
        Up = self.Vp + e
    
        # Choice is the combination that maximises U_p
        argmaxU = Up.argmax(axis=1)
        y_p = (Up == Up.max(axis=1,keepdims=True))
        y = self.combinations[argmaxU,:]

        # Compute the components of the Logit formula
        ev = np.exp(self.Vp)
        sev = np.sum(ev,axis=1,keepdims=True)

        # choice probs are the logit formula
        p = ev/sev

        # Compute log-likelihood
        ll = np.log(p**y_p)

        # Return choices and log-likelihood
        return y, ll
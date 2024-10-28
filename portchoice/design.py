"""Functions to create experimental designs for portfolio choice models."""
# Experimental design functions

# Load modules
from portchoice.expressions import Attribute
from portchoice.design_utils import _condgen, _initdesign, _maxcorr, _swapalg_portchoice

from locale import normalize
import numpy as np
import time
import datetime
import re
# from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# PortDesign class
class PortDesign:
    """Experimental design class

    This is the class to create experimental designs for portfolio choice
    models.

    Parameters
    ----------
    ATTLIST : list
        list that contains the specification of each attribute of the 
        experiment. Each element is an attribute object.
    NCS : int
        Number of choice situations.
    """
    def __init__(self, attlist: list, ncs: int):

        # Define scalars
        self.attlist = attlist
        self.N = ncs
        self.natts = len(self.attlist)

        # Get variable names of the design matrix
        self.names = [k.alt_name for k in self.attlist]
        self.levs = [k.levels for k in self.attlist]

    def gen_initdesign(self, cond: list = None, seed: bool = None):
        """Generate initial design matrix

        It generates the initial design matrix. The user can define a set of
        conditions that must be satisfied.

        Parameters
        -------
        cond : list[str], optional
            List of conditions that the final design must hold. Each element 
            is a string that contains a single condition. Conditions 
            can be of the form of binary relations (e.g., `X > Y` where `X` 
            and `Y` are attributes of a specific alternative) or conditional 
            relations (e.g., `if X > a then Y < b` where `a` and `b` are values).
            Users can specify multiple conditions when the operator `if` is defined, 
            separated by the operator `&`, by default None
        seed : bool, None
            Random seed, by default None

        Returns
        -------
        init_design : pandas.DataFrame
            A Pandas DataFrame with the initial design matrix.
        """

        # Generate conditions if defined
        if cond is not None:
            self.initconds = _condgen('desmat',cond,self.names,init=True)
            self.algconds = _condgen('swapdes',cond,self.names,init=False)
        else:
            self.initconds = None
            self.algconds = None

        # Set random seed if defined
        if seed is not None:
            np.random.seed(seed)

        # Generate initial design matrix
        init_design = _initdesign(levs=self.levs,ncs=self.N,cond=self.initconds)

        return pd.DataFrame(init_design,columns=self.names)

    def optimise(self, init_design: pd.DataFrame, iter_lim: int = None, noimprov_lim: int = None, time_lim: int = None, seed: int = None, verbose: bool = False):
        """Generate experimental design

        It generates an experimental design based on the specification 
        of the parent class and the parameters provided by the user.

        Parameters
        ----------
        initial_design : pandas.DataFrame
            The initial design matrix as a Pandas DataFrame
        iter_lim : int, optional
            Iteration limit, by default None
        noimprov_lim : int, optional
            Limit of iterations without improvement, by default None
        time_lim : float, optional
            Time limit in minutes, by default None
        seed : int, optional
            Random seed, by default None
        verbose : bool, optional
            Whether the algorithm prints output while optimising, by default True

        Returns
        -------
        optimal_design : np.ndarray
            Optimal design
        init_perf : float
            Value of the efficiency criterion at the **first** iteration
        final_perf : float
            Value of the efficiency criterion at the **last** iteration
        final_iter : int
            Total number of iterations
        """
        # Set random seed if defined
        if seed is not None:
            np.random.seed(seed)

        # Set stopping criteria if defined
        if iter_lim is None:
            iter_lim = np.inf
        
        if noimprov_lim is None:
            noimprov_lim = np.inf
            
        if time_lim is None:
            time_lim = np.inf

        ############################################################
        ########## Step 1: Set initial design performance ##########
        ############################################################

        if verbose:
            print('Evaluating initial design')

        desmat = init_design

        init_perf = _maxcorr(desmat)

        ############################################################
        ############## Step 2: Initialize algorighm ################
        ############################################################

        # Execute Swapping algorithm
        optimal_design, final_perf, final_iter, elapsed_time = _swapalg_portchoice(
            desmat,init_perf,self.algconds,iter_lim,noimprov_lim,time_lim)


        ############################################################
        ############## Step 3: Arange final design #################
        ############################################################

        # Add CS column
        optimal_design = np.c_[np.arange(self.N)+1,optimal_design]

        # Create Pandas DataFrame
        optimal_design = pd.DataFrame(optimal_design,columns=['CS'] + self.names)

        # Return a summary if verbose is True
        if verbose:
            print('Optimization complete')
            print('Elapsed time: ' + str(datetime.timedelta(seconds=elapsed_time))[:7])
            print('Performance of initial design: ',round(init_perf,6))
            print('Performance of last stored design: ',round(final_perf,6))
            print('Algorithm iterations: ',final_iter)
            print('')
        
        # Return the optimal design
        return optimal_design, init_perf, final_perf, final_iter

    # @staticmethod
    # def checkatts(ATTLIST: list, NCS: int):
        
    #     NGOODS = len(ATTLIST)

    #     # Check if there are enough goods defined.
    #     assert NGOODS >= 2, "Error: at least two goods must be specified."

    #     # Start check loop among goods
    #     for k in range(NGOODS):

    #         # Check if the element k is a dictionary
    #         assert isinstance(ATTLIST[k],dict), "Error: element that defines good " + str(k+1) + " is not a dictionary."

    #         # Start check loop among elements of each good
    #         for key, value in ATTLIST[k].items():

    #             # Check if each element is a list
    #             assert isinstance(value, list), "Error: Attribute \'" + key + "\'" + " in good " + str(k+1) + " is not a list."

    #             # Check if NCS is divisible by number of attribute levels
    #             assert NCS%len(ATTLIST[k][key]) == 0, "Error: No. of Choice sets is not divisible by number of levels of attribute \'" + key + "\'" + " in good" + str(k+1) + "."
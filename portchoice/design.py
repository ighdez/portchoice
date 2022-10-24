"""Functions to create experimental designs for portfolio choice models."""
# Experimental design functions

# Load modules
from locale import normalize
import numpy as np
import time
import datetime
import re
from sklearn.preprocessing import MinMaxScaler
from pandas import get_dummies

# PortDesign class
class PortDesign:
    """Experimental design class

    This is the class to create experimental designs for portfolio choice
    models.

    Parameters
    ----------
    ATTLIST : list
        List that contains the specification of each alternative of the 
        experiment. Each element is a dictionary, in which each element 
        is an attribute. The keys of each element are the attribute names 
        and each value is a list that contains the attribute levels.
    NCS : int
        Number of choice situations.
    """
    def __init__(self, ATTLIST: list, NCS: int):
        # Define scalars
        self.ATTLIST = ATTLIST
        self.NCS = NCS
        self.NGOODS = len(self.ATTLIST)

        # Run integrity checks on the attribute list
        PortDesign.checkatts(ATTLIST,NCS)

        # Get variable names of the design matrix
        self.NAMELIST = []
        self.LEVS = []

        for  k in range(self.NGOODS):
            for key, value in self.ATTLIST[k].items():
                self.NAMELIST = self.NAMELIST  + [key + '_' + str(k+1)]
                self.LEVS = self.LEVS + [value]
        
        # Number of attributes
        self.NATT = len(self.NAMELIST)

    def generate_design(self,ITERLIM: int = None, NOIMPROVLIM: int = None, TIMELIM: float = None, CRIT: str = 'deff', COND: list = None, SEED: int = None, VERBOSE: bool = True):
        """Generate experimental design

        It generates an experimental design based on the specification 
        of the parent class and the parameters provided by the user.

        Parameters
        ----------
        ITERLIM : int, optional
            Iteration limit, by default None
        NOIMPROVLIM : int, optional
            Limit of iterations without improvement, by default None
        TIMELIM : float, optional
            Time limit in minutes, by default None
        CRIT : str, optional
            Efficiency criterion. If `maxcorr`, then the algorithm minimises 
            the maximum value of the correlation between attributes. If `deff`, 
            the algorithm maximises the D-efficiency of a linear model, by default 'deff'
        COND : list, optional
            Conditions between attributes of the design. See the examples for an overview
            , by default None
        SEED : int, optional
            Random seed, by default None
        VERBOSE : bool, optional
            Whether the algorithm prints output while optimising, by default True

        Returns
        -------
        bestdes : np.ndarray
            Optimal design
        perf : float
            Value of the efficiency criterion at the **first** iteration
        bestperf : float
            Value of the efficiency criterion at the **last** iteration
        elapsed_time : float
            Optimisation time
        best_t : int
            Number of iterations
        """
        # Set random seed if defined
        if SEED is not None:
            np.random.seed(SEED)

        # Generate conditions if defined
        if COND is not None:
            initconds = PortDesign.condgen('desmat',COND,self.NAMELIST,init=True)
            algconds = PortDesign.condgen('swapdes',COND,self.NAMELIST,init=False)
        else:
            initconds = None
            algconds = None

        # If conditions are not defined, set them to infinity
        if ITERLIM is not None:
            iterlim = ITERLIM
        else:
            iterlim = np.inf

        if NOIMPROVLIM is not None:
            noimprovlim = NOIMPROVLIM
        else:
            noimprovlim = np.inf

        if TIMELIM is not None:
            timelim = TIMELIM
        else:
            timelim = np.inf

        ############################################################
        ########## Step 1: Generate initial design matrix ##########
        ############################################################

        if VERBOSE:
            print('Generating the initial design matrix')

        desmat = PortDesign.initdesign(self.NAMELIST,self.LEVS,self.NATT,self.NGOODS,self.NCS,initconds)
        
        if CRIT == 'deff':
            raise ValueError("D-efficiency is not implemented yet")
            perf = PortDesign.deff(desmat,self.NATT,self.NGOODS,self.NCS)
        elif CRIT == 'maxcorr':
            perf = PortDesign.maxcorr(desmat,self.NATT,self.NGOODS,self.NCS)
        elif CRIT == 'maxcorr_wide':
            perf = PortDesign.maxcorr_wide(desmat,self.NATT,self.NGOODS,self.NCS)

        ############################################################
        ############## Step 2: Initialize algorighm ################
        ############################################################

        bestdes, bestperf, best_t, elapsed_time = PortDesign.swapalg(desmat,perf,self.NATT,self.NGOODS,self.NCS,algconds,CRIT,iterlim,noimprovlim,timelim)

        ############################################################
        ############## Step 3: Arange final design #################
        ############################################################

        # Reshape in long format
        bestdes = np.reshape(bestdes,(self.NCS*self.NGOODS,int(self.NATT/self.NGOODS)))

        # Add design and version rows
        grprow = np.repeat(np.arange(1,self.NCS+1),self.NGOODS)
        grprow.shape = (self.NCS*self.NGOODS,1)
        altrow = np.tile(np.arange(1,self.NGOODS+1),self.NCS)
        altrow.shape = (self.NCS*self.NGOODS,1)

        bestdes = np.c_[grprow,altrow,bestdes]

        # If verbose, then print output
        if VERBOSE:
            print('Optimization complete')
            print('Elapsed time: ' + str(datetime.timedelta(seconds=elapsed_time))[:7])
            print('Performance of initial design: ',round(perf,6))
            print('Performance of last stored design: ',round(bestperf,6))
            print('Algorithm iterations: ',best_t)
            print('')

        return bestdes, perf, bestperf, elapsed_time, best_t

    @staticmethod
    def checkatts(ATTLIST: list, NCS: int):
        
        NGOODS = len(ATTLIST)

        # Check if there are enough goods defined.
        assert NGOODS >= 2, "Error: at least two goods must be specified."

        # Start check loop among goods
        for k in range(NGOODS):

            # Check if the element k is a dictionary
            assert isinstance(ATTLIST[k],dict), "Error: element that defines good " + str(k+1) + " is not a dictionary."

            # Start check loop among elements of each good
            for key, value in ATTLIST[k].items():

                # Check if each element is a list
                assert isinstance(value, list), "Error: Attribute \'" + key + "\'" + " in good " + str(k+1) + " is not a list."

                # Check if NCS is divisible by number of attribute levels
                assert NCS%len(ATTLIST[k][key]) == 0, "Error: No. of Choice sets is not divisible by number of levels of attribute \'" + key + "\'" + " in good" + str(k+1) + "."

    @staticmethod
    def condgen(DESNAME: str, COND: list, NAMELIST: list, init: bool = False):
        
        # Match variable names with columns in the design matrix
        if init:
            DESCOLUMNS = [DESNAME + '[i,' + str(i) + ']' for i in range(len(NAMELIST))]
        else:
            DESCOLUMNS = [DESNAME + '[:,' + str(i) + ']' for i in range(len(NAMELIST))]

        # Create new list of conditions
        NEWCONDS = []

        for c in COND:
            # Take a copy of the string
            cc = c[:]

            # Replace names by design matrix columns
            for i in range(len(NAMELIST)):
                cc = cc.replace(NAMELIST[i],DESCOLUMNS[i])
            
            # If there is a conditional 'if' statement, then convert it to a logical 'or'
            if 'if' in cc:
                # Split the sting in the 'then' part
                if_part, then_part = cc.split('then')

                # Set the 'logical_not' operator in the 'if' part
                if_part = 'np.logical_not(' + if_part.replace('if','') + ')'

                # In case there are '&' in the 'then' part, convert them to '*'
                then_part = then_part.replace('&','*')

                # Merge if and then parts
                cc = 'np.logical_or(' + if_part + ',' + then_part + ')'
            
            # Finally, append condition to condition list
            NEWCONDS.append(cc)
        
        # ...and return the new conditions list
        return NEWCONDS

    @staticmethod
    def initdesign(NAMELIST: list, LEVS: list, NATT: int, NGOODS: int, NCS: int, COND: list):

        # Create and populate the initial design matrix
        desmat = []

        for k in range(NATT):
            col = np.repeat(LEVS[k],NCS/len(LEVS[k]))
            np.random.shuffle(col)
            desmat.append(col)
        
        desmat = np.array(desmat).T

        # Apply conditions if needed
        if COND is not None:
            for i in range(NCS):
                # Check if all conditions are satisfied. If not, do a big enough loop till all conditions are satisfied
                check_all = []

                for c in COND:
                    check_all.append(eval(c))

                check_all = np.all(check_all)

                if not check_all:
                    for _ in range(1000):
                        # Create a random vector of levels for the row in question
                        desmat[i,:] = np.array([np.random.choice(i) for i in LEVS])
                        
                        # Check if conditions are met with the new vector in the design.
                        check_all = []

                        for c in COND:
                            check_all.append(eval(c))

                        check_all = np.all(check_all)


                        # If so, break the loop and go for the next row
                        if check_all:
                            break
                
                # If after the big loop conditions are not met, then raise an error
                assert check_all, 'It is not possible to met all conditions in the initial design matrix.'
        
        # Return the design matrix
        return desmat

    @staticmethod
    def deff(DES,NATT,NGOODS,NCS):
        pass

    @staticmethod
    # Maximum correlation function
    def maxcorr(DES,NATT,NGOODS,NCS):
        X = np.reshape(DES,(NCS*NGOODS,int(NATT/NGOODS)))
        # X = DES.copy()
        cormat = np.corrcoef(X.T)
        maxcor = np.max(np.abs(cormat) - np.identity(len(cormat)))
        return(maxcor)

    @staticmethod
    # Maximum correlation function in wide form
    def maxcorr_wide(DES,NATT,NGOODS,NCS):
        # X = np.reshape(DES,(NCS*NGOODS,int(NATT/NGOODS)))
        X = DES.copy()
        cormat = np.corrcoef(X.T)
        maxcor = np.max(np.abs(cormat) - np.identity(len(cormat)))
        return(maxcor)

    @staticmethod
    def swapalg(DES,INITPERF,NATT,NGOODS,NCS,COND,CRIT,ITERLIM,NOIMPROVLIM,TIMELIM):
        
        # Lock design matrix
        desmat = DES.copy()

        # Start stopwatch
        t0 = time.time()
        t1 = time.time()

        difftime = 0

        # Initialize algorithm parameters
        i = np.random.choice(np.arange(DES.shape[1]))
        t = 0
        ni = 0
        iterperf = INITPERF
        newperf = INITPERF

        # Start algorithm
        while True:
            
            # Iteration No.
            t = t+1
            
            # If one stopping criterion is satisfied, break!
            if ni >= NOIMPROVLIM or t >= ITERLIM or (difftime)/60 >= TIMELIM:
                break
            
            # Take a random swap
            pairswap = np.random.choice(desmat.shape[0],2,replace=False)

            # Check if attribute levels differ
            check_difflevels = desmat[pairswap[0],i] != desmat[pairswap[1],i]

            # If attribute levels differ, do the swap and check for conditions (if defined)
            if check_difflevels:
                swapdes = desmat.copy()
                swapdes[pairswap[0],i] = desmat[pairswap[1],i]
                swapdes[pairswap[1],i] = desmat[pairswap[0],i]
                
                # Check if conditions are satisfied after a swap
                check_all = []
                
                # If conditions are defined, this section will check that are satisfied, and rewrite 'check_satisfied_conds' if neccesary
                if COND is not None:
                    for c in COND:
                        check_all.append(eval(c))

                    check_all = np.all(check_all)
                
                # If all conditions are satisfied, compute D-error
                if check_all:
                    if CRIT == 'deff':
                        newperf = PortDesign.deff(swapdes,NATT,NGOODS,NCS)
                    elif CRIT == 'maxcorr':
                        newperf = PortDesign.maxcorr(swapdes,NATT,NGOODS,NCS)

            # ...else if they do not differ, keep the D-error
            else:
                newperf = iterperf.copy()
                
            # If the swap made an improvement, keep the design and update progress bar
            if CRIT == 'deff':
                improved = newperf > iterperf
            elif CRIT == 'maxcorr':
                improved = newperf < iterperf

            if improved:
                desmat = swapdes.copy()
                iterperf = newperf.copy()
                ni = 0
                
                # Update progress bar
                print('Optimizing. Press ESC to stop. / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / Performance: ' + str(round(iterperf,6)),end='\r')
            
            # ...else, pass to a random attribute and increment the 'no improvement' counter by 1.
            else:
                i = np.random.choice(np.arange(DES.shape[1]))
                ni = ni+1
            
            # Update progress bar each second
            if (difftime)%1 < 0.1:
                print('Optimizing. Press ESC to stop. / ' + 'Elapsed: ' + str(datetime.timedelta(seconds=difftime))[:7] + ' / Performance: ' + str(round(iterperf,6)),end='\r',flush=True)
            
            t1 = time.time()
            difftime = t1-t0
        
        # Return optimal design plus efficiency
        return desmat, iterperf, t, difftime

    @staticmethod
    def orthcode(X):
        pass
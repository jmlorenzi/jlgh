"""
Module for building a lattice gas hamiltonian
"""

import numpy as np
from copy import deepcopy

class LGHconfig:
    def __init__(self,nvector,mvector,Edft):
        """Class that contains a surface configuration
        definition to be used with the LGH class

        :param n: integer multipliers for fixed parameters
        :type n: list
        :param m: integer multipliers for free parameters
        :type m: list
        :param Edft: DFT energy for the case
        :type n: float
        """
        self.nvector = np.array([int(n) for n in nvector])
        self.mvector = np.array([int(m) for m in mvector])
        self.Edft = Edft


class LGH:
    def __init__(self,base=0.0,fixed=[],free=[],config_list=[]):
        self.base=base
        self.fixed=np.array(fixed)
        self.lfixed = len(self.fixed)
        self.free=np.array(free)
        self.lfree = len(self.free)
        
        if isinstance(config_list,list):
            for ic,config in enumerate(config_list):
                if not isinstance(config,LGHconfig):
                    raise TypeError("Configuration %s not right type" % ic)
            self.config_list=config_list                    
        else:
            raise TypeError("config_list must be a list")

    def energy(self,i,x):
        return self.base + np.dot(self.config_list[i].nvector,self.fixed) + \
            np.dot(self.config_list[i].mvector,x)

    def err(self,x):
        chisq = 0.0
        for ic, conf in enumerate(self.config_list):
            chisq+=(self.energy(ic,x)-conf.Edft)**2
        return chisq
    
    def derr(self,x):
        dchisq = np.zeros(len(self.free))
        for ic,conf in enumerate(self.config_list):
            for ip in xrange(self.lfree):
                dchisq[ip] += 2*(self.energy(ic,x)-conf.Edft)*x[ip]
        return dchisq


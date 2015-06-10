"""
Module for building a lattice gas hamiltonian
"""

import numpy as np
from copy import deepcopy

class LGHconfig:
    def __init__(self,nvector,mvector,Edft,name=None):
        """Class that contains surface configuration
        definition to be used with the LGH class. It
        is not a proper definition of the configuration,
        but only a list of the number of interactions
        that are present in that case.



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
        self.name = name


class LGH:
    """
    Class that defines a generic lattice gas hamiltonian.
    It contains a list of configurations which are instances of the
    LGHconfig class.

    The energy of a given configuration in this scheme is calculated
    as

    E(config) = base + \sum config.nvector[i] * fixed[i] +
    \sum config.mvector[i] * free[i]

    Where fixed and free represent energy contributions for the different
    clusters considered in the expansion. base is the energy of system
    with no adsorbates.
    """
    def __init__(self,base=0.0,fixed=[],free=[],config_list=[]):
        self.base=base
        self.fixed=np.array(fixed)
        self.lfixed = len(self.fixed)
        self.free=np.array(free)
        self.lfree = len(self.free)

        if isinstance(config_list,list):
            for ic,config in enumerate(config_list):
                if not isinstance(config,LGHconfig):
                    raise TypeError("Configuration nr %s not right type\n" % ic
                                    "Found type %s" % type(config))
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

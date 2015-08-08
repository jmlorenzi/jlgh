"""
Module for building a lattice gas hamiltonian
"""

import numpy as np
import copy
import scipy.optimize
import time

class LGHconfig:
    def __init__(self,nvector,mvector,Eref,name=''):
        """Class that contains a description of a surface
        configuration based on the interactions that are present
        to be used with the LGH class. It is but a list of
        the number of each interactions which are
        are present in that case under consideration.

        :param n: integer multipliers for fixed parameters
        :type n: list
        :param m: integer multipliers for free parameters
        :type m: list
        :param Eref: DFT energy for the case
        :type n: float
        """
        self.nvector = np.array([int(n) for n in nvector])
        self.mvector = np.array([int(m) for m in mvector])
        self.Eref = Eref
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
    def __init__(self,base=0.0,fixed=[],free=[],
                 ncells=1,config_list=[],
                 fixnames=[],freenames=[]):
        self.base=base
        self.fixed = np.array(fixed)
        self.lfixed = len(self.fixed)
        self.free = np.array(free)
        self.free0 = copy.copy(self.free)
        self.lfree = len(self.free)
        self.fixnames = fixnames
        self.freenames = freenames
        if self.fixnames:
            if not len(self.fixnames) == self.lfixed:
                        raise ValueError('fixnames of wrong size!')
        if self.freenames:
            if not len(self.freenames) == self.lfree:
                        raise ValueError('freenames of wrong size!')

        if isinstance(config_list,list):
            for ic,config in enumerate(config_list):
                if not isinstance(config,LGHconfig):
                    raise TypeError(("Configuration nr %s not right type\n" % ic)
                                    ("Found type %s" % type(config)))
                if not len(config.nvector) == self.lfixed:
                    raise ValueError("Configuration {:d}'s nvector".format(ic)+
                                     " is not of correct size")
                if not len(config.mvector) == self.lfree:
                    raise ValueError("Configuration {:d}'s mvector".format(ic)+
                                     " is not of correct size")
            self.config_list=config_list
        else:
            raise TypeError("config_list must be a list")

    def energy(self,i,x=None):
        if x is None:
            x = self.free
        return self.base + np.dot(self.config_list[i].nvector,self.fixed) + \
            np.dot(self.config_list[i].mvector,x)

    def err(self,x=None):
        chisq = 0.0
        for ic, conf in enumerate(self.config_list):
            chisq+=(self.energy(ic,x)-conf.Eref)**2
        return chisq

    def derr(self,x=None):
        dchisq = np.zeros(len(self.free))
        for ic,conf in enumerate(self.config_list):
            for ip in xrange(self.lfree):
                dchisq[ip] += 2*(self.energy(ic,x)-conf.Eref)*x[ip]
        return dchisq

    def maxerror(self):
        maerr = 0
        merr = 0
        imerr = -1
        for ic, config in enumerate(self.config_list):
            err = self.energy(ic) - config.Eref
            if abs(err) > maerr:
                maerr=abs(err)
                merr = err
                imerr = ic
        return merr, imerr

    def maxepa(self):
        """ Calculate the maximum error per adsorbate"""
        maepa = 0
        mepa = 0
        imepa = -1
        for ic, config in enumerate(self.config_list):
            epa = (self.energy(ic) - config.Eref) / float(sum(config.nvector))
            if abs(epa) > maepa:
                maepa=abs(epa)
                mepa = epa
                imepa = ic
        return mepa, imepa


    def print_fixpars(self):
        print('Base energy: %s' % self.base)
        print('Fixed parameters:')
        if self.fixnames:
            print('| '+'|'.join(['{:12}'.format(name) for name in
                           self.fixnames])+' |')
        print('| '+' |'.join('{:>+12.5f}'.format(val) for val in self.fixed)+' |')


    def print_freepars(self):
        print('Free parameters:')
        if self.freenames:
            print('| '+'|'.join('{:12}'.format(name) for name in
                           self.freenames)+' |')
        print('| '+ '|'.join('{:>+12.8f}'.format(val) for val in self.free)
               +' |')


    def optimize(self,method = 'Nelder-Mead', tol = 0.001,verbose=0):
        t0 = time.time()
        res = scipy.optimize.minimize(self.err,self.free,
                                      method=method,tol=tol)
        if res.success:
            self.free = res.x
            if verbose>1:
                print('Optimizing LGH using %s method' % method)
                print('Successfully reached minimum')
                print('after %i function evaluations' % res.nfev)
                for ic, config in enumerate(self.config_list):
                    if verbose > 2:
                        fmt = '  Config {:>3d} {:20} : Err = {:< g} , Epa = {:< g}'
                        Err = self.energy(ic)-config.Eref
                        Epa = Err / float(sum(config.nvector))
                        print(fmt.format(ic,config.name,Err,Epa))

                maxerr, iconf = self.maxerror()
                print('Maximum error: {} (conf. {})'.format(maxerr,iconf))
                maxerr, iconf = self.maxepa()
                print('Max. err. per ads.: {} (conf. {})'.format(maxerr,iconf))
                self.print_fixpars()
            if verbose:
                self.print_freepars()
                maxerr, iconf = self.maxerror()
                print('Maxerror : {} (conf {})'.format(maxerr,iconf))
        else:
            print('Could not converge!!')

    def print_error_analysis(self, to_stdout = True):
        """Analysis of the error of the LGH compared to DFT and to neglected LIs
        """
        message = ''
        message += 'Error report for LGH\n'
        message += '|{:5}|{:20}|{:15}|{:8}|{:8}|{:8}|{:8}|\n'.format(
            'Nr','Name','E_DFT','Err_LGH','per_ads','Err_noLI','per_ads')

        merr_LGH = 0.0
        sumerrLGH = 0.0
        imerr_LGH = 0

        merr_noLI = 0.0
        sumerrnoLI = 0.0
        imerr_noLI = 0

        mepa_LGH = 0.0
        sumepaLGH = 0.0
        imepa_LGH = 0

        mepa_noLI = 0.0
        sumepanoLI = 0.0
        imepa_noLI = 0

        tot_nads = 0

        for ic, config in enumerate(self.config_list):
            Err_LGH = self.energy(ic)-config.Eref
            Err_noLI = (self.energy(ic,np.zeros(len(self.free)))-config.Eref)
            nads = config.nvector.sum()
            message +='|{:5}|{:20}|{:15.03f}|{:8.03f}|{:8.03f}|{:8.03f}|{:8.03f}|\n'.format(
                ic,
                config.name,
                config.Eref,
                Err_LGH,
                Err_LGH / nads,
                Err_noLI,
                Err_noLI / nads,
                )
            if abs(Err_LGH) > abs(merr_LGH):
                merr_LGH = Err_LGH
                imerr_LGH = ic
            if abs(Err_LGH/nads) > abs(mepa_LGH):
                mepa_LGH = Err_LGH / nads
                imepa_LGH = ic
            if abs(Err_noLI) > abs(merr_noLI):
                merr_noLI = Err_noLI
                imerr_noLI = ic
            if abs(Err_noLI/nads) > abs(mepa_noLI):
                mepa_noLI = Err_noLI / nads
                imepa_noLI = ic

            sumerrLGH += abs(Err_LGH)
            sumerrnoLI += abs(Err_noLI)
            tot_nads += nads



        message += 'Maximum LGH error             : {:03f} (conf {})\n'.format(merr_LGH,imerr_LGH)
        message += 'Maximum LGH err. per adsorbate: {:03f} (conf {})\n'.format(mepa_LGH,imepa_LGH)
        message += 'Avg. LGH error             : {:03f}\n'.format(sumerrLGH / (ic+1))
        message += 'Avg. LGH err. per adsorbate: {:03f}\n'.format(sumerrLGH / tot_nads)
        message += 'Maximum noLI error            : {:03f} (conf {})\n'.format(merr_noLI,imerr_noLI)
        message += 'Maximum noLI er. per adsorbate: {:03f} (conf {})\n'.format(mepa_noLI,imepa_noLI)
        message += 'Avg. noLI error             : {:03f}\n'.format(sumerrnoLI / (ic+1))
        message += 'Avg. noLI err. per adsorbate: {:03f}\n'.format(sumerrnoLI / tot_nads)

        if to_stdout:
            print(message)
        return message


def loadfile(filename):
    """
    Parses and input file and returns the corresponding LGH object
    loaded with all the included configs
    """
    with open(filename,'r') as fin:
        lines = fin.readlines()

    fixnames = []
    freenames = []

    config_list = []
    for line in [L.strip() for L in lines if L.strip() != '']:
        if not line[0]=='#':
            if 'Ebase' in line:
                Ebase = float(line.split(':')[1].strip())
            elif 'fixnames' in line.lower():
                fixnames = line.split(':')[1].strip().split()
            elif 'freenames' in line.lower():
                freenames = line.split(':')[1].strip().split()
            elif 'fixed' in line.lower():
                fixed = [float(val) for val in
                         line.split(':')[1].strip().split()]
            elif 'free' in line.lower():
                free = [float(val) for val in
                         line.split(':')[1].strip().split()]
            else:
                name = ''
                entry = line.split(':')
                if len(entry)==3:
                    name = entry[0].strip()
                Eref = float(entry[-2])
                values = [int(val) for val in entry[-1].split()]
                nvector = values[:len(fixed)]
                mvector = values[len(fixed):]
                config_list.append(LGHconfig(nvector=nvector,mvector=mvector,
                                             Eref=Eref,name=name))
    return LGH(base=Ebase,fixed = fixed, free = free,
               config_list=config_list,
               fixnames=fixnames,
               freenames=freenames)

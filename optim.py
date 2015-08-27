import numpy as np
import scipy

from jlgh.types import LGH, Config

class LGHOptimizer(object):
    """
    The object that optimizes the LGH!!
    """
    def __init__(self,
                 LGH,
                 opt_surf_ene = False,
                 opt_bindings = False,
                 # exclude_species = None,
                 # exclude_clusters = None,
                 exclude_confs = None,
                 ):

        self.lgh = LGH
        self.opt_surf_ene = opt_surf_ene
        self.opt_bindings = opt_bindings

        if exclude_confs:
            self.config_list = [config for i,config in
                                enumerate(self.lgh.config_list)
                                if i not in exclude_confs]
        else:
            self.config_list = self.lgh.config_list

        self.nconf = len(self.config_list)


        e0_list = []
        a_list = []

        if opt_surf_ene and not opt_bindings:
            raise ValueError('opt_surf_ene must be False if'
                             ' opt_bindings is False')

        fix_cluster_enes = np.array([ self.lgh.cluster_energies[i] for i in
                                      self.lgh.fixed_cluster_indices])
        free_cluster_enes = [ self.lgh.cluster_energies[i] for i in
                                       xrange(self.lgh.nclusters) if not
                                      (i in self.lgh.fixed_cluster_indices)]
        for conf in self.config_list:
            fix_counts = np.array([conf.cluster_counts[i] for i in
                                   self.lgh.fixed_cluster_indices])
            free_counts = [conf.cluster_counts[i] for i
                           in xrange(self.lgh.nclusters) if not
                           (i in self.lgh.fixed_cluster_indices)]
            a_conf = []
            e0_conf = 0.0
            if not opt_surf_ene:
                e0_conf += self.lgh.base_energy*np.prod(conf.size)
            else:
                 a_conf.append(np.prod(conf.size))
            if not opt_bindings:
                # print('spec_counts shape')
                # print(conf.species_counts.shape)
                # print('be shape')
                # print(self.lgh.binding_energies.shape)

                e0_conf += np.dot(self.lgh.binding_energies,
                                  conf.species_counts)
            else:
                a_conf.extend(list(conf.species_counts))

            e0_conf += np.dot(fix_counts,fix_cluster_enes)
            a_conf.extend(free_counts)

            e0_list.append(e0_conf)
            a_list.append(a_conf)

        x = []
        if opt_surf_ene:
            x.append(self.lgh.base_energy)
        if opt_bindings:
            x.extend(list(self.lgh.binding_energies))
        x.extend(free_cluster_enes)

        # if opt_surf_ene and opt_bindings:
        #     for conf in self.lgh.config_list:
        #         e0_list.append( sum(
        #             [ conf.cluster_counts[i]*self.lgh.cluster_energies[i]
        #               for i in self.lgh.fixed_cluster_indices]))
        #         free_cluster_counts = [ic_count[1] for ic_count
        #                                in enumerate(conf.cluster_counts)
        #                                if ic_count[0] not
        #                                in self.lgh.fixed_cluster_indices]
        #         a_list.append(np.concatenate( ([np.prod(conf.size),],
        #                                        conf.species_counts,
        #                                        np.array(free_cluster_counts)))
        #     self.x = np.concatenate( ([self.lgh.base_energy,],
        #                             self.lgh.binding_energies,
        #                             self.lgh.cluster_energies,))
        # elif opt_bindings:
        #     for conf in self.lgh.config_list:
        #         e0_list.append(self.lgh.base_energy*np.prod(conf.size))
        #         a_list.append(np.concatenate( (conf.species_counts,
        #                                            conf.cluster_counts )))
        #     self.x = np.concatenate( (self.lgh.binding_energies,
        #                                  self.lgh.cluster_energies,))
        # elif not any([opt_bindings,opt_surf_ene]):
        #     for conf in self.lgh.config_list:
        #         e0_list.append(self.lgh.base_energy*np.prod(conf.size) +
        #                       np.dot(self.lgh.binding_energies,conf.species_counts))
        #         a_list.append(conf.cluster_counts)
        #     self.x = self.lgh.cluster_energies.copy()
        # else:
        #     raise NotImplementedError('lalal')

        self.e0 = np.array(e0_list)
        self.a = np.array(a_list)
        self.x = np.array(x)
        # print(self.e0)
        # print(self.a)
        # raise SystemExit()

    def err(self,x):
        chisq = 0.
        for i in xrange(self.nconf):
            chisq += ((self.e0[i] + np.dot(self.a[i],x)) \
                      - self.config_list[i].eref)**2
        return chisq

    def run(self,method = 'Nelder-Mead', tol = 0.001,verbose=0):
        x0 = self.x.copy()
        res = scipy.optimize.minimize(self.err,x0,
                                      method=method,tol=tol)
        if res.success:
            self.update_lgh(res.x)
            if verbose:
                print('Optimizing LGH using %s method' % method)
                print('Successfully reached minimum')
                print('after %i function evaluations' % res.nfev)
                self.lgh.print_cluster_energies()
            #     for ic, config in enumerate(self.lgh.config_list):
            #         if verbose > 2:
            #             fmt = '  Config {:>3d} {:20} : Err = {:< g} , Epa = {:< g}'
            #             Err = config.e - config.Eref
            #             Epa = Err / float(sum(config.nvector))
            #             print(fmt.format(ic,config.name,Err,Epa))

            #     # maxerr, iconf = self.maxerror()
            #     # print('Maximum error: {} (conf. {})'.format(maxerr,iconf))
            #     # maxerr, iconf = self.maxepa()
            #     # print('Max. err. per ads.: {} (conf. {})'.format(maxerr,iconf))
            #     # self.print_fixpars()
            # if verbose:
            #     self.lgh.print_status()
            #     # maxerr, iconf = self.maxerror()
            #     # print('Maxerror : {} (conf {})'.format(maxerr,iconf))
        else:
            print('Could not converge!!')
        return res.success

    def update_lgh(self,x):
        """ Update the lgh whith the result of optimization
        """
        icop = 0
        if self.opt_surf_ene:
            self.lgh.base_energy = x[0]
            icop +=1
        if self.opt_bindings:
            self.lgh.binding_energies = x[icop:][:self.lgh.nspecies]
            icop += self.lgh.nspecies
        for ie in xrange(self.lgh.nclusters):
            if not ie in self.lgh.fixed_cluster_indices:
                self.lgh.cluster_energies[ie] = x[icop]
                icop+=1


def cross_validation_score(lgh,tol = 1e-5, opt_surf_ene = False, opt_bindings = False):
    """ Calculates the Cross Validation (CV) score for a given lgh"""
    score = 0.0
    initial_guess = lgh.cluster_energies.copy()

    for iconf,conf in enumerate(lgh.config_list):
        lgh.cluster_energies = initial_guess.copy()
        opt = LGHOptimizer(lgh,
                           exclude_confs=[iconf,],
                           opt_surf_ene=opt_surf_ene,
                           opt_bindings=opt_bindings)
        if opt.run(verbose = 0, tol = tol):
            score += (conf.get_energy()-conf.eref)**2
        else:
            print('Optimization failed for configuration {}'.format(iconf))
            # print(conf)
            # raise RuntimeError()
    return np.sqrt(score/(iconf+1))

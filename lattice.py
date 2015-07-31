"""
Module used to convert lattice deffinitions to
the objects used to define the LGH
"""

import ase
import ase.lattice.surface

DIM = 2
DELTA_H_SMALL

class UnitCell(object):
    """ Class that contains the basic unit cell for the skeleton structure of the LGH.
    In our standard use case, this means the surface
    """
    def __init__(self, atoms, sites_pos=None, sites_names=None
                 h0 = None):

        self.atoms = atoms
        self.cell = atoms.cell[:DIM]

        if sites_pos:
            for isite, site_pos in enumerate(sites_pos):
                if len(site_pos) != DIM:
                    raise ValueError('Wrong positon dimension for site nr {:2d}'.format(isite))

        self.nsites = len(sites_pos)

        if sites_names:
            if len(sites_names) != self.nsites:
                raise ValueError('Number of names does not match number of sites')
            for site_name in sites_names:
                if not isinstance(site_name, str):
                    raise TypeError('Sites names must be strings')
            self.sites_names = sites_names
        else:
            self.sites_names = ['site{:02d}'.format(j) for j in xrange(len(self.nsites))]

        if h0:
            self.h0 = h0
        else:
            self.h0 = max([at.position[2] for at in self.atoms]) + DELTA_H_SMALL



class Adsorbate(object):
    """
    Defines an adsorbate as to be used in the building and interpreting of
    configurations.
    """
    def __init__(self, name,  atoms, origin_index = 0,
                 center_pos = [0.,0.,0.], def_height = None, max_radius = None):
        """ Initialize an Adsorbate instance

        Args:
        name (str) : The name of the adsorbate (must be unique!)

        atoms (ase.atoms.Atoms) : This defines standard (initial) geometry of the species.

        origin_index (int) : The index of the atom to be considered as 'central'. Defaults to 0.

        center_pos (list[float]) : The position of the 'center' of the molecule with
            respect to the 'center' molecule. Defaults to [0.,0.,0.]

        def_height (float) : The (initial) height at which the center of  adsorbate
            is to be placed by default.

        max_radius (float) : The maximum radius for which the components of the molecule are to be
            considered bound. Optional, defaults to None, which should be interpreted as inf.
        """

class Configuration(object):
    """
    Defines a configuration instance
    """

    TOL = 1e-5
    def __init__(self,dim=None,adsorbates=None,adsorbates_positions = None, atoms=None):
        pass

    def return_atoms(self):
        pass

    def __eq__(self,other):
        pass

    # def identify_clusters(self):



class Lattice(object):
    def __init__(self,unit_cell,size):
        self.dim = unit_cell.dim
        if isinstance( size, list):
            if len(cell) != dim:
                raise ValueError('Lattice size and dim incompatible')
            for x in size:
                if not isinstance(x, int):
                    raise TypeError('Lattice size must be integer')
            self.size = size
        elif isinstance(size, ( int) ):
            self.size = ( size,) * 3
        else:
            raise TypeError('size of wront type')

        self.volume = unit_cell.nsites
        for i in range(self.dim):
            self.volume *= size[i]

    def lattice2number(coord):
        ###

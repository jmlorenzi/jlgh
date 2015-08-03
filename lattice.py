"""
Module used to convert lattice definitions to
the objects used to define the LGH
"""

import ase
import ase.lattice.surface
import numpy as np


DIM = 2  # We use this as a parameter, but probably will always be fixed
DELTA_H_SMALL = 1.0 # Default height over the surface to place adsorbates

zvect = np.array([0.,0.,1.])

class UnitCell(object):
    """ Class that contains the basic unit cell for the skeleton structure of the LGH.
    In our standard use case, this means the surface
    """
    def __init__(self, atoms, sites_pos=None, sites_names=None,h0 = None):

        self.atoms = atoms
        self.cell = atoms.cell[:DIM]

        # Define the default height for the slab
        if h0:
            self.h0 = h0
        else:
            self.h0 = max([at.position[2] for at in self.atoms]) + DELTA_H_SMALL

        self.sites_pos = []
        if sites_pos:
            for isite, site_pos in enumerate(sites_pos):
                if len(site_pos) == 2:
                    self.sites_pos.append(site_pos[0]*self.cell[0] +
                                          site_pos[1]*self.cell[1] +
                                          self.h0*zvect)
                elif len(site_pos) == 3:
                    self.sites_pos.append(np.array(site_pos))
                else:
                    raise ValueError('Wrong positon for site nr {:2d}'.format(isite))

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

class Adsorbate(object):
    """
    Defines an adsorbate as to be used in the building and interpreting of
    configurations.
    """
    def __init__(self, name,  atoms, center = 0, max_radius = None):
        """ Initialize an Adsorbate instance

        Args:
        name (str) : The name of the adsorbate (must be unique!)

        atoms (ase.atoms.Atoms) : This defines standard (initial) geometry of the species.

        center (array-like or int) : The position of the 'center' of the molecule. It can either be
            an triplet or the index of the 'center' molecule. Defaults to 0.

        max_radius (float) : The maximum radius for which the components of the molecule are to be
            considered bound. Optional, defaults to None, which should be interpreted as inf.
        """
        self.name = name
        if isinstance(center, int):
            self.x0 = atoms.get_positions()[center]
        elif len(center) == 3:
            self.x0 = np.array(center)
        else:
            raise ValueError('Bad value of center')

        positions = atoms.get_positions()
        for pos in positions:
            pos -= self.x0

        atoms.set_positions(positions)
        self.atoms = atoms

        if max_radius is not None:
            self.max_radius = max_radius

    def __eq__(self,other):
        if isinstance(other,Adsorbate) and other.name = self.name:
            return True
        else:
            return False

class LGH(object):
    """
    Deifinition of the LGH

    The main class for this package, will override the one in __init__ when fully
    implmeneted
    """
    def __init__(self,**kwargs):
        attributes = ['adsorbate_list','cluster_list','config_list']
        for key, value in kwargs.items():
            if key in attributes:
                setattr(self, key, value)

    def add_adsorbate(self,adsorbate):
        pass

    def add_cluster(self,cluster):
        pass

    def add_config(self,config):
        pass


class Configuration(object):
    """
    Defines a configuration instance
    """

    TOL = 1e-5
    def __init__(self,
                 unit_cell=None,
                 size=None,
                 adsorbates_coords=None,
                 atoms=None):
        """
        Initialize a Configuration instance

        Args
        ----

        unit_cell : Instance of the UnitCell class that defines the surface

        size (array-like[int]) : Dimensions of the unit cell defining the configuration

        adsorbates ( list [ (Adsorbate, (int, int, str/int)) ] ) : Defines the coverage of
            the Configuration
        """
        if atoms is None:
            if not all([unit_cell,size,adsorbates_coords]):
                raise RuntimeError('unit_cell, size, and adsorbates_coords are needed'
                                   'to define a Configuration')
            self.unit_cell = unit_cell
            if not len(size) == DIM:
                raise ValueError('Wrong dimension for size!!')
            self.size = size
            self.adsorbates_coords = adsorbates_coords
        else:
            raise NotImplementedError('Detecting configurations from atoms'
                                      ' not implemented yet!')

        self.species_list = sorted(list(set([ads_coord[0].name
                                             for ads_coord in
                                             self.adsorbates_coords])))
        self.nspecs = len(self.species_list)

        matrix = np.zeros([size[0],size[1],self.unit_cell.nsites],int)
        for ads_coord in self.adsorbates_coords:
            if isinstance(ads_coord[1][2],str):
                isite = self.unit_cell.sites_list.index(ads_coord[1][2])
            else:
                isite = ads_coord[1][2]
            if matrix[ads_coord[1][0],ads_coord[1][1],isite]:
                raise ValueError('Site {},{},{} assigned twice!'.format(
                    ads_coord[1][0],ads_coord[1][1],isite))
            else:
                matrix[ads_coord[1][0],ads_coord[1][1],isite] =
                self.species_list.index(ads_coord[0])




    def return_atoms(self):
        """Builds the atoms object that corresponds to the configuration
        """
        atoms = self.unit_cell.atoms * [self.size[0],self.size[1],1]

        for adsorbate, coord in self.adsorbates_coords:
            if isinstance(coord[2],str):
                isite = self.unit_cell.sites_names.index(coord[2])
            else:
                isite = coord[2]


            x0cell = (coord[0]*self.unit_cell.cell[0]
                      + coord[1]*self.unit_cell.cell[1])
                      #+ self.unit_cell.h0*zvect)

            ads_positions = adsorbate.atoms.get_positions()
            for ads_pos in ads_positions:
                # print(type(x0cell))
                # # print(x0cell.shape)
                # print(x0cell)
                # print(type(self.unit_cell.sites_pos[isite]))
                # # print(self.unit_cell.sites_pos[isite].shape)
                # print(self.unit_cell.sites_pos[isite])
                # print(type(ads_pos))
                # # print(ads_pos.shape)
                # print(ads_pos)

                ads_pos += x0cell + self.unit_cell.sites_pos[isite]

                print(ads_pos)
            print(ads_positions)
            toadd = adsorbate.atoms.copy()
            toadd.set_positions(ads_positions)
            atoms += toadd
        return atoms

    def get_cluster_multiplicity(self,cluster):
        """
        Returns the number of times a cluster is repeated in the configuration

        All cluster that have at leaste one of its participating sites within
        the Configuration unit cell are counted

        Args
        ---

        cluster (Cluster) : Instance of the Cluster class
        """

        NREP = 1 # TODO make update with cluster size

        for ix in xrange(-NREP,NREP+1):
            for iy in xrange(-NREP,NREP+1):
                for instance in cluster.instance_list:







    def __eq__(self,other):
        ## TODO build this
        return False


class Cluster(object):
    """
    Defines a cluster class to be included in a cluster expansion

    For now, symmetry is NOT automatically taken into account
    """

    def __init__(self,name,instances_list):
        """
        Initialize a cluster class

        Args
        ---
        name (str) : identifing name of the cluster
        instances_list (list[(adsorbate,[dx,dy,site])]) : list of sets of
            (relative) coordinates that define the cluster
        """
        self.name = name
        self.instances_list = instances_list





# class Lattice(object):
#     def __init__(self,unit_cell,size):
#         self.dim = unit_cell.dim
#         if isinstance( size, list):
#             if len(cell) != dim:
#                 raise ValueError('Lattice size and dim incompatible')
#             for x in size:
#                 if not isinstance(x, int):
#                     raise TypeError('Lattice size must be integer')
#             self.size = size
#         elif isinstance(size, ( int) ):
#             self.size = ( size,) * 3
#         else:
#             raise TypeError('size of wront type')

#         self.volume = unit_cell.nsites
#         for i in range(self.dim):
#             self.volume *= size[i]

#     def lattice2number(coord):
#         pass

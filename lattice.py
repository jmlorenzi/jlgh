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

class LGH(object):
    """
    Contains the definition of the LGH, including the set of all posible
    adsorbates, the set of all cluster and all the configurations to be
    fitted to the cluster expansion model
    """

    def __init__(self,**kwargs):
        if 'sucell' in kwargs.keys():
            self.sucell = kwargs['sucell']

        self.adsorbate_list = []
        self.config_list = []
        self.clustergoup_list = []

    def add_sucell(self,sucell):
        """Define the unit cell for the surface"""
        if not hasattr(self,'sucell'):
            self.sucell = sucell

    def add_adsorbate(self,adsorbate):
        """Add an adsorbate"""
        self.adsorbate_list.append(adsorbate)

    def add_adsorbates(self,ads_list):
        """Add a list of adsorbates"""
        self.adsorbate_list.extend(ads_list)

    def add_config(self,config):
        """Add a configuration"""
        config.set_lgh(self)
        self.config_list.append(config)

    def add_configs(self,config_list):
        """Add a list of configurations"""
        for config in config_list:
            config.set_lgh(self)
            self.config_list.append(config)

    def add_clustergroup(self,clustergroup):
        """Add a Cluster Group"""
        self.clustergoup_list.append(clustergroup)

    def add_clustergroups(self,clustergroup_list):
        """Add a Cluster Group"""
        self.clustergoup_list.extend(clustergroup_list)

    def get_atoms(self,iconf):
        return self.config_list[iconf].get_atoms()

    def identify_clusters(self,iconf):
        return self.config_list[iconf].identify_clusters()

    def get_species(self):
        return sorted([ads.name for ads in self.adsorbate_list])


class SurfUnitCell(object):
    """ Class that contains the basic unit cell for the skeleton structure of the LGH.
    In our standard use case, this means the surface
    """
    def __init__(self, atoms, sites_list=None):

        self.atoms = atoms

        self.cell = atoms.cell
        self.nsites = len(sites_list)

        self.sites_list = sorted(sites_list,key=lambda site: site.name)

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

class Site(object):
    """Class that defines a site in the SurfaceUnitCell.
    Heavily inspired in the kmos.types.Site class.
    Thanks to Max Hoffmann for coding that
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get('name', '')
        self.tags = kwargs.get('tags', '')
        if 'pos' in kwargs:
            if type(kwargs['pos']) is str:
                self.pos = np.array([float(i) for i in kwargs['pos'].split()])
            elif type(kwargs['pos']) in [np.ndarray, tuple, list]:
                self.pos = np.array(kwargs['pos'])
            else:
                raise Exception('Input %s not understood!' % kwargs['pos'])
        else:
            self.pos = np.array([0., 0., 0.])


    def __repr__(self):
        return '[SITE] {0:12s} {2:s} {3:s}'.format(self.name,
                                                   self.pos,
                                                   self.tags)

class BareCoord(object):
    """
    A bare Coord definition to be included in Config class instances
    """
    def __init__(self,sitename,offset):
        self.name = sitename
        if type(offset) in [ list, tuple, np.ndarray]:
            if len(offset) == 1:
                self.offset = np.array([offset[0], 0, 0], int)
            elif len(offset) == 2:
                self.offset = np.array([offset[0], offset[1], 0], int)
            elif len(offset) == 3:
                self.offset = np.array([offset[0],
                                        offset[1],
                                        offset[2]], int)
            else:
                raise ValueError('Wrong size for offset')
        else:
            raise TypeError('Offset type not supported')


def parse_spec_coord(sc_string):
    spec, terms = sc_string.split('@')

    term = terms.split('.')
    if len(term) == 2:
        print(term[1])
        print(type(term[1]))
        coord = BareCoord(term[0].strip(),eval(term[1]),
                      )
    elif len(term) == 1:
        coord = BareCoord(term[0].strip(),(0, 0, 0),)
    else:
        raise ValueError("Cannot parse coord description")

    return spec, coord


class Config(object):
    """
    Defines a configuration instance
    """

    TOL = 1e-5
    def __init__(self,
                 size,
                 species_coords,
                 ):
        """
        Initialize a Configuration instance

        Args
        ----

        size (array-like[int]) : Dimensions of the unit cell defining the configuration

        adsorbates ( list [ (name, bare_coord) ] ) : Defines how the surface is filled
        """
        if not len(size) == DIM:
            raise ValueError('Wrong dimension for size!!')

        self.size = size
        # Check consistency of species_coords

        self.species_coords = []

        for i, spec_coord in enumerate(species_coords):
            if isinstance(spec_coord, str):
                spec, coord = parse_spec_coord(spec_coord)
            else:
                spec, coord = spec_coord
            for i in xrange(DIM):
                if coord.offset[i] >= DIM:
                    raise NotImplementedError(
                'Coordinate {0}{1} falls outside configuration'.format(
                    coord.name,coord.offset))
            self.species_coords.append((spec,coord))

    def set_lgh(self,lgh):
        """
        Link the LGH with the current configuration.

        Srveral of the other methods only work with this set
        """
        self.lgh = lgh

    def get_species(self):
        return sorted(list(set([spec_coord[0]
                    for spec_coord in self.species_coords])))

    def get_sites(self):
        return sorted(list(set([spec_coord[1].name
                    for spec_coord in self.species_coords])))

    def get_multiplicity(self):
        multip = []
        for spec in self.get_species():
            multip.append(len([spec_coord for spec_coord in
                               self.species_coords if spec_coord[0] == spec]))
        return np.array(multip)

    def get_coverages(self):
        surf = 1
        for i in xrange(DIM):
            surf *= self.size[i]
        return self.get_multiplicity() / float(surf)

    def return_atoms(self):
        """Builds the atoms object that corresponds to the configuration
        """
        atoms = self.lgh.sucell.atoms * [self.size[0],self.size[1],1]

        for species, coord in self.species_coords:
            adsorbate = [ads for ads in self.lgh.adsorbate_list
                          if ads.name == species][0]
            site = [ssite for ssite in self.lgh.sucell.sites_list
                    if ssite.name == coord.name][0]

            rcoord = ((coord.offset[0]+site.pos[0])*self.lgh.sucell.cell[0]
                     + (coord.offset[1]+site.pos[1])*self.lgh.sucell.cell[1]
                     +  site.pos[2]*self.lgh.sucell.cell[2])

            ads_positions = adsorbate.atoms.get_positions()
            for ads_pos in ads_positions:
                ads_pos += rcoord

            toadd = adsorbate.atoms.copy()
            toadd.set_positions(ads_positions)
            atoms += toadd
        return atoms

    def calculate_matrix(self):

        if not self.lgh:
            species_list = self.get_species()
            sites_list = self.get_sites()
        else:
            species_list = self.lgh.get_species()
            sites_list = [s.name for s in self.lgh.sucell.sites_list]

        matrix = np.zeros([self.size[0],self.size[1],len(sites_list)],int)

        # Fill up the matrix
        for spec, coord in self.species_coords:
            matrix[coord.offset[0],
                   coord.offset[1],
                   sites_list.index(coord.name)] = (species_list.index(spec) + 1)

        self.matrix = matrix

    def identify_clusters(self):
        """ Count the number of repetitions for each cluster in the LGH
        """
        if not hasattr(self,'matrix'):
            self.calculate_matrix()
        species_list = self.lgh.get_species()
        sites_list = [s.name for s in self.lgh.sucell.sites_list]

        count = [0,]*len(self.lgh.clustergoup_list)
        # Go through all the surface
        for ix in xrange(self.size[0]):
            for iy in xrange(self.size[1]):
                # and all cluster
                for icg, cluster_group in enumerate(self.lgh.clustergoup_list):
                    for cluster in cluster_group.clusters:
                        # and finally all coordinates
                        for species, coord in cluster.species_coords:
                            # relative coordinates wrapped back to
                            # the config unit cell
                            xrel = (ix + coord.offset[0]) % self.size[0]
                            yrel = (iy + coord.offset[1]) % self.size[1]
                            if not ( self.matrix[xrel,
                                               yrel,
                                               sites_list.index(coord.name)]
                                == (species_list.index(species) + 1 )):
                                break
                        else:
                            print('Found match for cluster {}'.format(cluster_group.name))
                            print('In position {},{}'.format(ix,iy))
                            count[icg] += 1
        for icg, cluster_group in enumerate(self.lgh.clustergoup_list):
            print('Count for cluster {0} = {1:d}'.format(cluster_group.name,count[icg]))



    def __eq__(self,other):
        ## TODO build this
        return False


class Cluster(object):
    """
    Defines and individual cluster, without any consideration for symmetry
    """
    def __init__(self,species_coords):
        self.species_coords = []
        for spec_coord in species_coords:
            if isinstance(spec_coord,str):
                spec, coord = parse_spec_coord(spec_coord)
            else:
                spec, coord = spec_coord
            self.species_coords.append((spec,coord))

class ClusterGroup(object):
    """
    Class that holds a cluster group. That is all clusters that are equivalent through
    symmetry transformations
    """
    def __init__(self, name, energy, cluster_list):
        self.name = name
        self.energy = energy
        self.clusters = cluster_list

"""
Module used to convert lattice definitions to
the objects used to define the LGH
"""

import ase
import ase.lattice.surface
import numpy as np
from jlgh.tools import *
from struformat.molcrys import cluster

DIM = 2  # We use this as a parameter, but probably will always be fixed
DELTA_H_SMALL = 1.0 # Default height over the surface to place adsorbates

TOL_ANGLE = 1e-9
TOL_DIST = 1e-6

zvect = np.array([0.,0.,1.])

def get_dot_cross_angle(v1,v2):
    """
    Wraps the operations of geting angles
    between vectors to reduce code repetition
    """
    cross = np.cross(v1,v2)
    dot = np.dot(v1,v2)
    angle = np.arctan2(np.linalg.norm(cross),dot)
    return dot, cross, angle

class LGH(object):
    """
    Contains the definition of the LGH, including the set of all posible
    adsorbates, the set of all cluster and all the configurations to be
    fitted to the cluster expansion model
    """

    def __init__(self,**kwargs):
        if 'base_cell' in kwargs.keys():
            self.base_cell = kwargs['base_cell']
            self.cell = self.base_cell.cell

        self.adsorbate_list = []
        self.config_list = []
        self.clustergoup_list = []

        self.base_energy = None
        self.binding_energies = None
        self.cluster_energies = None

    def add_base_cell(self,base_cell):
        """Define the unit cell for the surface"""
        if not hasattr(self,'base_cell'):
            self.base_cell = base_cell
            self.cell = base_cell.cell

    def add_adsorbate(self,adsorbate):
        """Add an adsorbate"""
        self.adsorbate_list.append(adsorbate)

    def add_adsorbates(self,ads_list):
        """Add a list of adsorbates"""
        self.adsorbate_list.extend(ads_list)

    def add_config(self,config):
        """Add a configuration"""
        config.set_lgh(self)
        config.update_species_counts()
        config.identify_clusters()
        self.config_list.append(config)

    def add_configs(self,config_list):
        """Add a list of configurations"""
        for config in config_list:
            self.add_config(config)

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

    def get_species_names(self):
        return sorted([ads.name for ads in self.adsorbate_list])

    def get_adsorbates(self):
        return sorted([ads for ads in self.adsorbate_list],
                      key = lambda x:x.name)
    def reset(self):
        """
        Cleans up the previous calculation and prepares
        all arrays for a new one
        """

        self.base_energy = self.base_cell.eini
        binding_energies = []
        for ads in self.get_adsorbates:
            binding_energies.append(ads.eini)
        self.binding_energies = np.array(binding_energies)

        cluster_energies = []
        for cluster in self.clustergroup_list:
            cluster_energies.append(cluster.eini)
        self.cluster_energies = np.array(cluster_energies)

        for conf in self.config_list:
            conf.update_species_counts()
            conf.identify_clusters()

    def get_energy(self,iconf):
        return self.config_list[iconf].get_energy()

    def err(self):
        chisq = 0.0
        for config in self.config_list:
            chisq += (config.get_energy() - config.eref)**2
        return chisq

    def read_atoms(self,atoms):
        """ Takes an atoms object and identifies the
        corresponding configuration associated to it
        Returns False in failure

        Note: Only works for units cell with z axis
        perpendicular to the xy plane
        """
        # We do not want to mess with the original atoms
        natoms = atoms.copy()

        self._rotate_to_elementary_cell(natoms)

        size = self._get_atoms_size(natoms)
        print('Found that atoms match a ({0}x{1}) unit cell'.format(*size))

        self._remove_surface(natoms,size)

        # spliting geometry using C. Schober's tool
        fragments = cluster(natoms)

        # And sort out which fragments are where adsorbed
        config_description = self._sort_fragments(size,natoms,fragments)

        conf = Config(size,config_description)
        print('Identified configuration')
        print(conf)
        self.add_config(conf)

        return True

    def _rotate_to_elementary_cell(self,natoms):
        """ Rotates an atom object so its vectors
        are parallel to those of the elementary unit cell.

        Returns False when the unite cell vector directions
        do not match
        """
        # We can first use the unit cell vectors angles
        for pairs in [(0,1,),(1,2,),(2,0)]:
            _, _, a_cell = get_dot_cross_angle(self.cell[pairs[0]],
                                               self.cell[pairs[1]])
            _, _, a_atoms = get_dot_cross_angle(natoms.cell[pairs[0]],
                                                natoms.cell[pairs[1]])
            if abs(a_cell - a_atoms) > TOL_ANGLE:
                print('Angles between axes {} and {} for atoms cell'
                      ' differ from the one in the elementary cell')
                return False

        # Rotate atoms so its x-axis matches that from base_cell
        xdot, xnormal, angle = get_dot_cross_angle(natoms.cell[0],self.cell[0])
        if angle > TOL_ANGLE:
            print('Rotating {} rad to match x axes'.format(angle))
            natoms.rotate(xnormal / np.linalg.norm(xnormal), angle, rotate_cell = True)

        # Project y axes into the x axis
        xversor = self.cell[0] / np.linalg.norm(self.cell[0])
        y_cell_aux = np.dot(self.cell[1],xversor)*xversor
        y_atoms_aux = np.dot(natoms.cell[1],xversor)*xversor
        # And use this to get the perpendicular part
        y_cell_aux = self.cell[1] - y_cell_aux
        y_atoms_aux = natoms.cell[1] - y_atoms_aux
        # And the angle between them
        _, xparall, angle = get_dot_cross_angle(y_atoms_aux, y_cell_aux,)
        # if we did things right xparall should be true to its name
        _, _, zero_angle = get_dot_cross_angle(xparall,xversor)
        if min(abs(zero_angle),abs(zero_angle-np.pi)) > TOL_ANGLE:
            print('zero_angle = {}'.format(zero_angle))
            raise RuntimeError('Should never come here! zero_angle not zero!')

        # Perform the final rotation!
        if abs(angle) > TOL_ANGLE:
            print('Rotating {} rad around x axis'.format(angle))
            natoms.rotate(xparall / np.linalg.norm(xparall), angle, rotate_cell = True)

        # And explicity again check all angles
        for i in xrange(3):
            _, _, angle = get_dot_cross_angle(natoms.cell[i],self.cell[i])
            if abs(angle) > TOL_ANGLE:
                raise RuntimeError(
            'Something went wrong! Angle of axes {} do not match'.format(i))

        return True

    def _get_atoms_size(self,natoms):
        """
        Returns the size of the atoms objects as compared
        to the elementary unit cell
        """

        fnx = np.linalg.norm(natoms.cell[0]) \
                 / np.linalg.norm(self.cell[0])
        nx = round(fnx)
        if abs(fnx-nx) > TOL_DIST:
            print('Size of x vector of atoms cell is too off')
            return False
        nx = int(nx)

        fny = np.linalg.norm(natoms.cell[1]) \
                 / np.linalg.norm(self.cell[1])
        ny = round(fny)
        if abs(fny-ny) > TOL_DIST:
            print('Size of y vector of atoms cell is too off')
            return False
        ny = int(ny)

        return (nx,ny)

    def _remove_surface(self,natoms,size):
        """
        Tries to match the predicted surface to the
        surface contained in the atoms object and removes
        those atoms from it, so as to facilitate the
        recognition of adsorbates
        """

        bare_surf = self.base_cell.atoms * [size[0],size[1],1]

        max_min_dist = 0.0 # Maximun value of the minumum distance
                           # between atoms of natoms and bare_surf
        for nat in natoms:
            dists = [(sat.position[2] - nat.position[2]) for sat in
                      bare_surf if nat.symbol == sat.symbol]
            if dists:
                dz_min = min(dists, key = lambda x:abs(x))
            else:
                continue
            if abs(dz_min) > abs(max_min_dist):
                max_min_dist = dz_min

        if abs(max_min_dist) > TOL_DIST:
            print("Correcting surface for z displacement by {0} A".format(
                                                        max_min_dist))
            natoms.translate(np.array([0.,0.,max_min_dist]))

        # As a failure check we will enforce that atoms constrained
        # in the elemetary cell need match very well the atoms in the
        # case being tested
        const_ind = []
        if bare_surf.constraints:
            for const in bare_surf.constraints:
                const_ind.extend(list(const.index))

        from ase.visualize import view

        for isat, sat in enumerate(bare_surf):
            num_dists = [ (i,np.linalg.norm(sat.position-nat.position)) for
                          i,nat in enumerate(natoms) if nat.symbol == sat.symbol]
            imin, dmin = min(num_dists, key = lambda x:x[1])
            if (isat in const_ind) and (dmin > TOL_DIST):
                raise ValueError('Constrained substrate atoms do not'
                                 ' match surface structure by {}'.format(dmin))
            natoms.pop(imin)

    def _sort_fragments(self,size,natoms,fragments):
        """ Sorts a list of ase.Atoms objects (fragments)
        into configurations on the defined surface
        """
        frag_symbs = [frag.get_chemical_symbols() for frag in fragments]
        ads_symbs = [ads.atoms.get_chemical_symbols() for ads
                     in self.adsorbate_list]

        config_description = []
        for ifr, frag in enumerate(fragments):
            for iads, ads_symb in enumerate(ads_symbs):
                if (frag_symbs[ifr] == ads_symb):
                    break
            else:
                raise ValueError('Fragment {}'.format(frag.get_chemical_symbols())+
                                 ' not recognized')
            cmpos = frag.get_center_of_mass()
            # project center of mass to xy plane
            cmpos = project_to_plane(cmpos,natoms.cell[0],natoms.cell[1])
            # go through all sites to see which is closest
            sitesdists = []
            for ix in xrange(size[0]):
                for iy in xrange(size[1]):
                    for site in self.base_cell.sites_list:
                        pos = ix*self.cell[0] + \
                              iy*self.cell[1] + \
                              site.pos
                        pos = project_to_plane(pos,
                                               natoms.cell[0],
                                               natoms.cell[1])
                        dist = np.linalg.norm(pos-cmpos)
                        sitesdists.append(((ix,iy,site.name),dist))
            site = min(sitesdists,key = lambda x: x[1])
            descr = '{}@{}.({},{},0)'.format(self.adsorbate_list[iads].name,
                                             site[0][2],
                                             site[0][0],
                                             site[0][1])
            config_description.append(descr)

        return config_description

class BaseCell(object):
    """ Class that contains the basic unit cell for the skeleton structure of the LGH.
    In our standard use case, this means the surface
    """
    def __init__(self, atoms, sites_list=None, energy = None):

        self.atoms = atoms
        self.cell = atoms.cell
        self.nsites = len(sites_list)
        self.sites_list = sorted(sites_list,key=lambda site: site.name)
        self.eini = energy
        self.e = energy

class Adsorbate(object):
    """
    Defines an adsorbate as to be used in the building and interpreting of
    configurations.
    """
    def __init__(self, name,  atoms, center = 0,
                    binding_energy = None, max_radius = None):
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

        self.eini = binding_energy
        self.e = binding_energy

class Site(object):
    """Class that defines a site in the BaseCell.
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

    def __eq__(self,other):
        return all([self.name == other.name,
                   not any(self.offset - other.offset)])

    def __ne__(self,other):
        return any([self.name != other.name,
                    any(self.offset - other.offset)])

    def __repr__(self):
        return '[BARE_COORD] {0:12s} {1}'.format(self.name,self.offset)

def parse_spec_coord(sc_string):
    spec, terms = sc_string.split('@')

    term = terms.split('.')
    if len(term) == 2:
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
                 energy = None,
                 ):
        """
        Initialize a Configuration instance

        Args
        ----

        size (array-like[int]) : Dimensions of the unit cell defining the configuration

        adsorbates ( list ) : Either a list of (species_name,coord) pairs or of strings of
            the form 'species@site.(x,y,0)'
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

        self.species_counts = None
        self.cluster_counts = None
        self.eref = energy
        self.e = energy

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

    def get_species_count(self):
        if self.lgh:
            species_list = [spec.name for spec in self.lgh.get_species_names()]
        else:
            species_list = self.get_species()
        count = []
        for spec in species_list:
            count.append(len([spec_coord for spec_coord in
                               self.species_coords if spec_coord[0] == spec]))
        return np.array(count)

    def update_species_counts(self):
        self.species_counts = self.get_species_count()

    def get_coverages(self):
        surf = 1
        for i in xrange(DIM):
            surf *= self.size[i]
        return self.species_counts / float(surf)

    def return_atoms(self):
        """Builds the atoms object that corresponds to the configuration
        """
        atoms = self.lgh.base_cell.atoms * [self.size[0],self.size[1],1]

        for species, coord in self.species_coords:
            adsorbate = [ads for ads in self.lgh.adsorbate_list
                          if ads.name == species][0]
            site = [ssite for ssite in self.lgh.base_cell.sites_list
                    if ssite.name == coord.name][0]

            rcoord = ((coord.offset[0]+site.pos[0])*self.lgh.base_cell.cell[0]
                     + (coord.offset[1]+site.pos[1])*self.lgh.base_cell.cell[1]
                     +  site.pos[2]*self.lgh.base_cell.cell[2])

            ads_positions = adsorbate.atoms.get_positions()
            for ads_pos in ads_positions:
                ads_pos += rcoord

            toadd = adsorbate.atoms.copy()
            toadd.set_positions(ads_positions)
            atoms += toadd
        return atoms

    def calculate_matrix(self):
        """
        Prepares a reperesentation of the configuration in matrix form,
        which facilitates identifying clusters
        """

        if not self.lgh:
            species_list = self.get_species()
            sites_list = self.get_sites()
        else:
            species_list = self.lgh.get_species_names()
            sites_list = [s.name for s in self.lgh.base_cell.sites_list]

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
        species_list = self.lgh.get_species_names()
        sites_list = [s.name for s in self.lgh.base_cell.sites_list]

        count = [0,]*len(self.lgh.clustergoup_list)
        # Go through all the surface
        for ix in xrange(self.size[0]):
            for iy in xrange(self.size[1]):
                # and all cluster
                for icg, cluster_group in enumerate(self.lgh.clustergoup_list):
                    for cluster in cluster_group.clusters:
                        # and finally all coordinates
                        for cent_spec, cent_coord in cluster.species_coords:
                            # twice, since we need to account for shifts
                            if not ( self.matrix[ix,iy,
                                    sites_list.index(cent_coord.name)] ==
                                    (species_list.index(cent_spec)+1)):
                                continue
                            # print('Conf Coord {},{}'.format(ix,iy))
                            # print('Cent Coord name: {}'.format(cent_coord.name))
                            # print('Cent Coord offset: {},{}'.format(cent_coord.offset[0],
                            #                                         cent_coord.offset[1]))
                            # print(len([ x for x in
                            #     cluster.species_coords if x[1] != cent_coord]))

                            for spec, coord in [ x for x in
                                cluster.species_coords if x[1] != cent_coord]:
                                # get coordinates relative to the center,
                                # folded into the central copy
                                xrel = (ix + coord.offset[0] - cent_coord.offset[0])\
                                  % self.size[0]
                                yrel = (iy + coord.offset[1] - cent_coord.offset[1])\
                                  % self.size[1]
                                if not ( self.matrix[xrel,
                                                   yrel,
                                                   sites_list.index(coord.name)]
                                    == (species_list.index(spec) + 1 )):
                                    # print('Skipped')
                                    break
                            else:
                                # print('Found match for cluster {}'.format(cluster_group.name))
                                # print('In position {},{}'.format(ix,iy))
                                count[icg] += 1
                                # print('Counted one for {}'.format(cluster_group.name))
        self.counts = np.array(count)
        # for icg, cluster_group in enumerate(self.lgh.clustergoup_list):
        #     print('Count for cluster {0} = {1:d}'.format(cluster_group.name,count[icg]))

    def get_energy(self):
        if not self.lgh:
            raise UserWarning('Need a lgh to calculate energy')
            return False

        ebase = self.lgh.base_energy
        return self.lgh.base_energy*np.prod(self.size) \
               + np.dot(self.species_counts,self.lgh.binding_energies) \
               + np.dot(self.cluster_counts,self.lgh.cluster_energies)

    def __eq__(self,other):
        ## TODO build this
        return False

    def __repr__(self):
        rep = '[CONF] ({0}x{1})\n'.format(*self.size)
        for spec, coord in self.species_coords:
            rep += '  {0}@{1}.({2},{3},0)\n'.format(spec,
                                                 coord.name,
                                                 coord.offset[0],
                                                 coord.offset[1])
        return rep

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

        self.eini = energy
        self.e = energy

        self.clusters = []
        for cluster_def in cluster_list:
            if isinstance(cluster_def,Cluster):
                self.clusters.append(cluster_def)
            else:
                self.clusters.append(Cluster(cluster_def))



class LGHOptimizer(object):
    """
    The object that optimizes the LGH!!
    """
    def __init__(self,LGH, opt_surf_ene = False, opt_bindings = False):

        self.lgh = LGH
        if opt_surf_ene and opt_bindings:
            fixed_list = []
            for conf in self.lgh.config_list:
                fixed_list.append(np.concatenate( ([np.prod(conf.size),],
                                              conf.species_counts,
                                              conf.cluster_counts )))
            self.fixed_array = np.array(fixed_list)

            self.free = np.concatenate( ([self.lgh.base_energy,],
                                    self.lgh.binding_energies,
                                    self.lgh.cluster_energies,))

        elif opt_bindings:
            fixoff = []
            fixed_list = []
            for conf in self.lgh.config_list:
                fixoff.append(self.lgh.base_energy*np.prod(conf.size))
                fixed_list.append(np.concatenate( (conf.species_counts,
                                                   conf.cluster_counts )))
            self.fixed_array = np.array(fixed_list)

            self.free = np.concatenate( (self.lgh.binding_energies,
                                         self.lgh.cluster_energies,))

        elif not any(opt_bindings,opt_surf_ene):
            fixoff = []
            fixed_list = []
            for conf in self.lgh.config_list:
                fixoff.append(self.lgh.base_energy*np.prod(conf.size) +
                              np.dot(self.lgh.binding_energy,conf.species_counts))
                fixed_list.append(conf.cluster_counts)

            self.fixed_array = np.array(fixed_list)

            self.free = self.lgh.cluster_energies.copy()

    def err(self,free):
        ####

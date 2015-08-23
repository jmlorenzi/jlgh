import ase
import ase.io
from ase.constraints import FixAtoms, FixedLine
import ase.lattice.surface
import numpy as np
from struformat.molcrys import cluster
import time, scipy, sys, os, glob

from jlgh import DIM, TOL_ANGLE, TOL_DIST
from jlgh.utils import get_dot_cross_angle, project_to_plane, nr2letter

class LGH(object):
    """
    Main class of the module, contains all other elements that help you
    define the Cluster Expansion.

    ``Attributes''
    -------------

        directory (str) : Directory where the LGH is saved

        base_cell (BaseCell) : Definition of the basic surface cell

        adsorbate_list (List(Adsorbate)) : List of all species

        clustergroup_list (List(ClusterGroup)) : List of all clusters

        config_list (List(Config)) : List of all configurations

        base_energy (float) : Energy of a 1x1 empty surface slab

        binding_energy (np.ndarray(float)) : List of the energies of
            the adsorbates in the ZCL (with respect to the clean surface,
            NOT the clean surface + gas phase)

        cluster_energies (np.ndarray(float)) : Energy contributions of
            each cluster

        nspecies (int) : Number of species

        nclusters (int) : Number of cluster groups

        nconfig (int) : Number of configurations

    ``Methods''
    -----------
        add_base_cell : add a BaseCell instance

        add_adsorbate : add an adsorbate

        add_adsorbates : add a list of adsorbates

        add_clustergroup : add a cluster group

        add_clustergroups : add a list of cluster groups

        add_config : Add a configuration and update its own methods
            to accomodate to this LGH

        add_configs : Add a list of configurations (and
            update each one accordingly)

        get_atoms : Construct and return an ase.Atoms object
            representing the one of the configurations

        _count_clusters : Calculate the multiplicity of clusters
            in one of the configurations, and save

        get_species_names : Get a list of adsorbates names in alphabetical
            order

        get_adsorbates : Get a list of adsorbates in alphabetical order

        reset : Update the LGH state to make it ready for an optimization.

        get_energy : Returns the (cluster expansion) energy corresponding
            to the selected configuration

        err : Return the error of the LGH

        read_atoms : Adds a configuration to the LGH from an atoms objects
    """

    def __init__(self,**kwargs):

        if 'name' in kwargs.keys():
            self.name = kwargs['name']
        else:
            self.name = 'lgh'

        if 'base_cell' in kwargs.keys():
            self.base_cell = kwargs['base_cell']

        if 'directory' in kwargs.keys():
            self.directory = os.path.abspath(kwargs['directory'])
        else:
            self.directory = os.path.abspath('')

        if 'name' in kwargs.keys():
            self.name = kwargs['name']

        self.adsorbate_list = []
        self.config_list = []
        self.clustergroup_list = []

        self.base_energy = None
        self.binding_energies = None
        self.cluster_energies = None

    def add_base_cell(self,base_cell):
        """Define the unit cell for the surface"""
        if not hasattr(self,'base_cell'):
            self.base_cell = base_cell

    def add_adsorbate(self,adsorbate):
        """Add an adsorbate"""
        self.adsorbate_list.append(adsorbate)

    def add_adsorbates(self,ads_list):
        """Add a list of adsorbates"""
        self.adsorbate_list.extend(ads_list)

    def add_config(self,config):
        """Add a configuration"""
        config.set_lgh(self)
        config._count_species()
        config._count_clusters()
        self.config_list.append(config)

    def add_configs(self,config_list):
        """Add a list of configurations"""
        for config in config_list:
            self.add_config(config)

    def add_clustergroup(self,clustergroup):
        """Add a Cluster Group"""
        self.clustergroup_list.append(clustergroup)

    def add_clustergroups(self,clustergroup_list):
        """Add a Cluster Group"""
        self.clustergroup_list.extend(clustergroup_list)

    def save(self,
             # cluster_fname = None,
             save_atoms = True,
             overwrite_atoms = True,
             ):
        """
        Save the Cluster Expansion definition and components
        """
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        general_outf_name = os.path.join(self.directory,
                                         '{}.lgh'.format(self.name))
        general_outf = open(general_outf_name,'w')

        general_outf.write('#'*72+'\n')
        general_outf.write('#  {0:68}#'.format('jlgh Cluster Expansion')+'\n')
        general_outf.write('#'*72+'\n')
        general_outf.write('name : {}\n'.format(self.name))
        general_outf.write('nsites : {}\n'.format(self.nsites))
        general_outf.write(('sites : ' +
                             ('{} '*self.nsites)).format(*self.get_site_names())
                             +'\n')
        general_outf.write('nspecies : {}\n'.format(self.nspecies))
        general_outf.write(('species : ' +
                             ('{} '*self.nspecies)).format(*self.get_species_names())
                             +'\n')
        general_outf.write('\n')
        # general_outf.close()

        # if cluster_fname is None:
        #     cluster_fname = os.path.join(self.directory,'clusters')
        # else:
        #     cluster_fname = os.path.abspath(cluster_fname)

        # cluster_file = open(cluster_fname,'w')

        cluster_file = general_outf

        cluster_file.write('nclusters : {}\n'.format(self.nclusters))
        cluster_file.write('cluster groups :'+'\n')
        for clustergroup in self.clustergroup_list:
            cluster_file.write( ('{} : '
                                 + '{} '*clustergroup.nclusters).format(
                                     clustergroup.name,
                                     *clustergroup.clusters)+
                                     '\n')

        cluster_file.write('\n')
        # cluster_file.close()

        # Save base_cell
        if not os.path.isdir(os.path.join(self.directory,'base')):
            os.mkdir(os.path.join(self.directory,'base'))

        base_trajf_name = os.path.join(self.directory,'base','base.traj')
        ase.io.write(base_trajf_name,self.base_cell.atoms) ###

        base_outf_name = os.path.join(self.directory,'base','base')
        base_outf = open(base_outf_name,'w')
        base_outf.write('# Base file \n')
        base_outf.write('sites: \n')
        for site in self.base_cell.site_list:
            base_outf.write('{} : {} \n'.format(site.name, site.pos))
        base_E0_name = os.path.join(self.directory,'base','E0')
        with open(base_E0_name,'w') as base_E0:
            base_E0.write('{0:.12f}\n'.format(self.base_cell.eini))

        # Save adsorbates
        for ads in self.get_adsorbates():
            ads_dir = os.path.join(self.directory,
                                    ads.name,)
            if not os.path.isdir(ads_dir):
                os.mkdir(ads_dir)
            ads_trajf_name = os.path.join(ads_dir,
                                           '{}.traj'.format(ads.name))
            ase.io.write(ads_trajf_name,ads.atoms)
            ads_E0_name = os.path.join(self.directory,
                                        ads.name,
                                        'E0')
            with open(ads_E0_name,'w') as ads_E0:
                ads_E0.write('{0:.12f}\n'.format(ads.eini))

        # Save configs
        config_file = general_outf
        config_file.write('nconf = {}\n'.format(self.nconf))
        for config in self.config_list:
            if not os.path.isdir(config.directory):
                os.makedirs(config.directory)
            config_trajf_name = os.path.join(config.directory,
                                             'initial.traj')
            if save_atoms:
                if any([overwrite_atoms,
                        not os.path.exists(config_trajf_name)]):
                    ase.io.write(config_trajf_name,
                                 config.return_atoms())
            with open(os.path.join(config.directory,'config'),'w') as fconf:
                fconf.write('{}'.format(config))
            with open(os.path.join(config.directory,'counts'),'w') as fcounts:
                fcounts.write('# ')
                fcounts.write(('{} ' * self.nclusters).format(
                                 *[x.name for x in self.clustergroup_list])
                                 + '\n')

                fcounts.write(('{:d} ' * self.nclusters).format(
                                 *config.cluster_counts) + '\n')
            if hasattr(config,'eref') and config.eref is not None:
                with open(os.path.join(config.directory,'eref'),'w') as feref:
                    feref.write('{}\n'.format(config.eref))
            config_file.write('{}\n'.format(config.directory))
        config_file.close()

    def load(self,
             name = 'lgh',
             scan_configs=True,
             ):
        """Loads a lgh configuration

           Arguments:
           ---------

           name (str): name of lgh to read

           scan_config (bool) : If true, configurations will be search
               in the default folders and loaded

        """
        pass

    def print_cluster_energies(self):
        """ Print a review of all clusters' energies"""
        for ic, cluster in enumerate(self.clustergroup_list):
            txt = '{0:20} : {1:01.03f}'.format(
                   cluster.name,
                   self.cluster_energies[ic])
            if cluster.fixed:
                txt = txt + '(f)'
            print(txt)

    def print_binding_energies(self, resp_gas = False):
        """ Prints the binding energy of the species,
        as used in the Cluster Expansion

        Parameters
        ----------
        resp_gas (bool) : If True, prints the binding energy with respect
        to the gas phase energy (not implemented, default False)
        """

        if resp_gas:
            raise NotImplementedError('No gas phase energy yet')

        ic = 0
        for spec in self.get_species_names():
            for site in self.get_site_names():
                print('E_{0}_{1} = {2}'.format(
                    spec,site,self.binding_energies[ic]))
                ic += 1

    def get_atoms(self,iconf):
        return self.config_list[iconf].get_atoms()

    def _count_clusters(self,iconf):
        return self.config_list[iconf]._count_clusters()

    def get_species_names(self):
        return sorted([ads.name for ads in self.adsorbate_list])

    def get_site_names(self):
        return [site.name for site in self.base_cell.site_list]

    def get_adsorbates(self):
        return sorted([ads for ads in self.adsorbate_list],
                      key = lambda x:x.name)

    def sort_configs(self):
        self.config_list = sorted(self.config_list,
                                  key = lambda x:(x.size[0],
                                                  x.size[1],
                                                  x.get_code(),
                                                  )
                                  )

    def reset(self):
        """
        Cleans up the previous calculation and prepares
        all arrays for a new one
        """
        self.nspecies = len(self.adsorbate_list)
        self.nsites = self.base_cell.nsites
        self.nconf = len(self.config_list)
        self.nclusters = len(self.clustergroup_list)

        self.base_energy = self.base_cell.eini
        binding_energies = []
        for ads in self.get_adsorbates():
            if type(ads.eini) in [int,float]:
                binding_energies.extend([ads.eini,]*self.nsites)
            elif type(ads.eini) in [list,tuple,np.ndarray]:
                if not len(ads.eni) == self.nsites:
                    raise ValueError(
                      'Number of binding energies for {0}'.format(ads.name) +
                      ' does not match number of sites')
                binding_energies.extend(list(ads.eini))

        self.binding_energies = np.array(binding_energies)

        fixed_cluster_indices = []
        cluster_energies = []
        for ic,cluster in enumerate(self.clustergroup_list):
            if cluster.fixed:
                fixed_cluster_indices.append(ic)
            cluster_energies.append(cluster.eini)

        # self.fixed_clusters = fixed_clusters
        # self.fixed_cluster_energies = np.array(fixed_cluster_energies)
        # self.free_clusters = free_clusters
        self.fixed_cluster_indices = fixed_cluster_indices
        self.cluster_energies = np.array(cluster_energies)

        # self.sort_configs()
        for conf in self.config_list:
            conf.update()

    def get_energy(self,iconf):
        return self.config_list[iconf].get_energy()

    def err(self):
        chisq = 0.0
        for config in self.config_list:
            chisq += (config.get_energy() - config.eref)**2
        return chisq

    def read_atoms(self,atoms,verbose = 0):
        """ Takes an atoms object and identifies the
        corresponding configuration associated to it
        Returns False in failure

        Note: Only works for units cell with z axis
        perpendicular to the xy plane
        """
        # We do not want to mess with the original atoms
        natoms = atoms.copy()

        done_rotation = self._rotate_to_elementary_cell(natoms, verbose=verbose)
        if not done_rotation:
            return False

        size = self._get_atoms_size(natoms)
        if not size:
            return False
        elif verbose:
            print('Found that atoms match a ({0}x{1}) unit cell'.format(*size))

        self._remove_surface(natoms,size, verbose=verbose)

        # spliting geometry using C. Schober's tool
        fragments = cluster(natoms)

        # And sort out which fragments are where adsorbed
        config_description = self._sort_fragments(size,natoms,fragments)

        conf = Config(size,config_description)
        if verbose:
            print('Identified configuration')
            print(conf)

        self.add_config(conf)
        return conf

    def _rotate_to_elementary_cell(self,natoms, verbose=0):
        """ Rotates an atom object so its vectors
        are parallel to those of the elementary unit cell.

        Returns False when the unite cell vector directions
        do not match
        """
        # We can first use the unit cell vectors angles
        for pairs in [(0,1,),(1,2,),(2,0)]:
            _, _, a_cell = get_dot_cross_angle(self.base_cell.cell[pairs[0]],
                                               self.base_cell.cell[pairs[1]])
            _, _, a_atoms = get_dot_cross_angle(natoms.cell[pairs[0]],
                                                natoms.cell[pairs[1]])
            if abs(a_cell - a_atoms) > TOL_ANGLE:
                print('Angles between axes {} and {} for atoms cell'
                      ' differ from the one in the elementary cell')
                return False

        # Rotate atoms so its x-axis matches that from base_cell
        xdot, xnormal, angle = get_dot_cross_angle(natoms.cell[0],self.base_cell.cell[0])
        if angle > TOL_ANGLE:
            print('Rotating {} rad to match x axes'.format(angle))
            natoms.rotate(xnormal / np.linalg.norm(xnormal), angle, rotate_cell = True)

        # Project y axes into the x axis
        xversor = self.base_cell.cell[0] / np.linalg.norm(self.base_cell.cell[0])
        y_cell_aux = np.dot(self.base_cell.cell[1],xversor)*xversor
        y_atoms_aux = np.dot(natoms.cell[1],xversor)*xversor
        # And use this to get the perpendicular part
        y_cell_aux = self.base_cell.cell[1] - y_cell_aux
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
            if verbose:
                print('Rotating {} rad around x axis'.format(angle))
            natoms.rotate(xparall / np.linalg.norm(xparall), angle, rotate_cell = True)

        # And explicity again check all angles
        for i in xrange(3):
            _, _, angle = get_dot_cross_angle(natoms.cell[i],self.base_cell.cell[i])
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
                 / np.linalg.norm(self.base_cell.cell[0])
        nx = round(fnx)
        if abs(fnx-nx) > TOL_DIST:
            print('Size of x vector of atoms cell is too off')
            return False
        nx = int(nx)

        fny = np.linalg.norm(natoms.cell[1]) \
                 / np.linalg.norm(self.base_cell.cell[1])
        ny = round(fny)
        if abs(fny-ny) > TOL_DIST:
            print('Size of y vector of atoms cell is too off')
            return False
        ny = int(ny)

        return (nx,ny)

    def _remove_surface(self, natoms, size, verbose=0):
        """
        Tries to match the predicted surface to the
        surface contained in the atoms object and removes
        those atoms from it, so as to facilitate the
        recognition of adsorbates
        """

        bare_surf = self.base_cell.atoms * [size[0],size[1],1]

        # As a failure check we will enforce that atoms constrained
        # in the elemetary cell need match very well the atoms in the
        # case being tested
        const_ind = []
        if bare_surf.constraints:
            for const in bare_surf.constraints:
                const_ind.extend(list(const.index))

        max_min_dist = 0.0 # Maximun value of the minumum distance
                           # between atoms of natoms and bare_surf
        for isat, sat in enumerate(bare_surf):
            if not isat in const_ind:
                continue
            dists = [(sat.position[2] - nat.position[2]) for nat in
                      natoms if nat.symbol == sat.symbol]
            if dists:
                dz_min = min(dists, key = lambda x:abs(x))
            else:
                continue
            if abs(dz_min) > abs(max_min_dist):
                max_min_dist = dz_min

        if abs(max_min_dist) > TOL_DIST:
            if verbose:
                print("Correcting surface for z displacement by {0} A".format(
                                                        max_min_dist))
            pos0 = natoms.get_positions()
            pos0 += np.array([0.,0.,max_min_dist])
            natoms.positions = pos0

        for isat, sat in enumerate(bare_surf):
            num_dists = [ (i,np.linalg.norm(sat.position-nat.position)) for
                          i,nat in enumerate(natoms) if nat.symbol == sat.symbol]
            imin, dmin = min(num_dists, key = lambda x:x[1])
            if (isat in const_ind) and (dmin > TOL_DIST):
                raise ValueError('Constrained substrate atoms do not'
                                 ' match surface structure by {} for at {}'.format(dmin,imin))
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
                    for site in self.base_cell.site_list:
                        pos = ix*self.base_cell.cell[0] + \
                              iy*self.base_cell.cell[1] + \
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
    def __init__(self, atoms, site_list=None, energy = None):

        self.atoms = atoms
        self.cell = atoms.cell
        self.nsites = len(site_list)
        self.site_list = sorted(site_list,key=lambda site: site.name)
        self.eini = energy
        self.e = energy

class Adsorbate(object):
    """
    Defines an adsorbate as to be used in the building and interpreting of
    configurations.
    """
    def __init__(self, name,  atoms, center = [0,0,0],
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

    def __eq__(self,other):
        return self.name == other.name
    def __ne__(self,other):
        return not self.__eq__(other)
    def __lt__(self,other):
        return self.name < other.name
    def __gt__(self,other):
        return not any([self == other, self < other])
    def __le__(self,other):
        return any([self == other, self < other])
    def __ge__(self,other):
        return any([self == other, self > other])


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

    def __eq__(self,other):
        return self.name == other.name

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
                if coord.offset[i] >= self.size[i]:
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
        if self.lgh:
            return self.lgh.get_species_names()
        else:
            return sorted(list(set([spec_coord[0]
                        for spec_coord in self.species_coords])))

    def get_sites(self):
        if self.lgh:
            return self.lgh.get_site_names()
        else:
            return sorted(list(set([spec_coord[1].name
                    for spec_coord in self.species_coords])))

    def _count_species(self):
        species_list = self.get_species()
        sites_list = self.get_sites()
        count = []
        for spec in species_list:
            for site in sites_list:
                count.append(len([spec_coord for spec_coord in
                                  self.species_coords if
                                  (spec_coord[0] == spec and
                                   spec_coord[1].name == site)]))
        self.species_counts = np.array(count)

    def get_coverages(self):
        surf = 1
        for i in xrange(DIM):
            surf *= self.size[i]
        return self.species_counts / float(surf)

    def update(self):
        """
        Update configuration's attributes

        Counts the number of species, the multiplicity of each defined
        cluster and saves the directory where the configuration will be
        stored
        """
        self._count_species()
        self._count_clusters()
        self.directory = os.path.join(self.lgh.directory,
                                      '{0:d}x{1:d}'.format(*self.size),
                                      self.get_code())

    def return_atoms(self):
        """Builds the atoms object that corresponds to the configuration
        """
        atoms = self.lgh.base_cell.atoms * [self.size[0],self.size[1],1]

        for species, coord in self.species_coords:
            adsorbate = [ads for ads in self.lgh.adsorbate_list
                          if ads.name == species][0]
            site = [ssite for ssite in self.lgh.base_cell.site_list
                    if ssite.name == coord.name][0]

            rcoord = ((coord.offset[0]+site.pos[0])*self.lgh.base_cell.cell[0]
                     + (coord.offset[1]+site.pos[1])*self.lgh.base_cell.cell[1]
                     +  site.pos[2]*self.lgh.base_cell.cell[2])

            ads_positions = adsorbate.atoms.get_positions()
            for ads_pos in ads_positions:
                ads_pos += rcoord

            toadd = adsorbate.atoms.copy()

            new_constraints = []
            if toadd.constraints:
                for constr in toadd.constraints:
                    if isinstance(constr, FixAtoms):
                        new_indices = []
                        for n, val in enumerate(constr.index):
                            if val.dtype.name.startswith('bool'):
                                if not val:
                                    continue
                                new_indices.append(n+len(atoms))
                            elif val.dtype.name.startswith('int'):
                                new_indices.append(val+len(atoms))
                        new_constraints.append(FixAtoms(indices = new_indices))
                    elif isinstance(constr, FixedLine):
                        new_constraints.append(FixedLine(
                                                a = constr.a+len(atoms),
                                                direction = constr.dir))
                    else:
                        raise NotImplementedError(
                          'Constraint of type {}'.format(type(costr)) +
                          ' not implemented yet for adsorbates')

            toadd.positions = ads_positions
            old_constraints = atoms._get_constraints()
            atoms += toadd
            atoms.set_constraint(old_constraints + new_constraints)
        return atoms

    def calculate_matrix(self):
        """
        Prepares a reperesentation of the configuration in matrix form,
        which facilitates identifying clusters
        """

        if not self.lgh:
            species_list = self.get_species()
            site_list = self.get_sites()
        else:
            species_list = self.lgh.get_species_names()
            site_list = [s.name for s in self.lgh.base_cell.site_list]

        matrix = np.zeros([self.size[0],self.size[1],len(site_list)],int)

        # Fill up the matrix
        for spec, coord in self.species_coords:
            matrix[coord.offset[0],
                   coord.offset[1],
                   site_list.index(coord.name)] = (species_list.index(spec) + 1)

        self.matrix = matrix

    def _count_clusters(self):
        """ Count the number of repetitions for each cluster in the LGH
        """
        if not hasattr(self,'matrix'):
            self.calculate_matrix()
        species_list = self.lgh.get_species_names()
        site_list = [s.name for s in self.lgh.base_cell.site_list]

        count = [0,]*len(self.lgh.clustergroup_list)
        # Go through all the surface
        for ix in xrange(self.size[0]):
            for iy in xrange(self.size[1]):
                # and all cluster
                for icg, cluster_group in enumerate(self.lgh.clustergroup_list):
                    for cluster in cluster_group.clusters:
                        # and finally all coordinates
                        for cent_spec, cent_coord in cluster.species_coords:
                            # twice, since we need to account for shifts
                            if not ( self.matrix[ix,iy,
                                    site_list.index(cent_coord.name)] ==
                                    (species_list.index(cent_spec)+1)):
                                continue
                            for spec, coord in [ x for x in
                                cluster.species_coords if x[1] != cent_coord]:
                                xrel = (ix + coord.offset[0] - cent_coord.offset[0])\
                                  % self.size[0]
                                yrel = (iy + coord.offset[1] - cent_coord.offset[1])\
                                  % self.size[1]
                                if not ( self.matrix[xrel,
                                                   yrel,
                                                   site_list.index(coord.name)]
                                    == (species_list.index(spec) + 1 )):
                                    break
                            else:
                                count[icg] += 1
        self.cluster_counts = np.array(count)

    def get_energy(self):
        if not self.lgh:
            raise UserWarning('Need a lgh to calculate energy')
            return False

        ebase = self.lgh.base_energy
        return self.lgh.base_energy*np.prod(self.size) \
               + np.dot(self.species_counts,self.lgh.binding_energies) \
               + np.dot(self.cluster_counts,self.lgh.cluster_energies)

    def update_energy(self):
        self.e = self.get_energy()

    def __eq__(self,other):
        if not isinstance(other,Config):
            raise NotImplementedError('Can only compare config to config')
        if not self.size == other.size:
            return False
        elif not any(self.matrix - other.matrix):
            return True
        else:
            return False

    def __ne__(self,other):
        return not self == other

    def __lt__(self,other):
        if not isinstance(other,Config):
            raise NotImplementedError('Can only compare config to config')
        if self.size < other.size:
            return True
        elif self.size > other.size:
            return False
        if (self.get_species_count().sum() <
            other.get_species_count().sum()):
            return True
        elif (self.get_species_count().sum() >
            other.get_species_count().sum()):
            return False
        return (list(self.matrix.flatten()) <
                list(other.matrix.flatteh()))

    def __gt__(self,other):
        return not any([self == other, self < other])

    def __le__(self,other):
        return any([self == other, self < other])

    def __ge__(self,other):
        return any([self == other, self > other])

    def __repr__(self):
        rep = '[CONF] ({0}x{1})\n'.format(*self.size)
        for spec, coord in self.species_coords:
            rep += '  {0}@{1}.({2},{3},0)\n'.format(spec,
                                                 coord.name,
                                                 coord.offset[0],
                                                 coord.offset[1])
        return rep

    def get_code(self):
        """
        Returns a code identifies the config in the lgh
        """

        ord_species_coords = sorted(self.species_coords,
            key = lambda x: (x[0],
                             x[1].offset[0],
                             x[1].offset[1],
                             x[1].name)
                                   )

        code = '_'.join([x[0] for x in ord_species_coords]) + '_'

        for coord in [x[1] for x in ord_species_coords]:
            code += nr2letter(coord.offset[0])
            code += nr2letter(coord.offset[1])
            code += nr2letter(self.lgh.get_site_names().index(coord.name))

        # for iy in xrange(self.size[1]):
        #     for ix in xrange(self.size[0]):
        #         for isite in xrange(self.lgh.nsites):
        #             n = self.matrix[ix,iy,isite]
        #             if n <= 9:
        #                 ch = str(n)
        #             elif n <= (ord('Z') - ord('A') + 10):
        #                 ch = chr(ord('A') + n - 10)
        #             else:
        #                 raise NotImplementedError('I ran out of letters'
        #                                           ' for the species...'
        #                                           ' 35 is too many...')
        #             code = ch + code
        return code

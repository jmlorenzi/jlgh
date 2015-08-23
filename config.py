from jlgh import DIM
import numpy as np
from ase.constraints import FixAtoms, FixedLine, FixCartesian
import os
from jlgh.lgh import parse_spec_coord
from jlgh.utils import nr2letter

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

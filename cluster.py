"""
Defines the Cluster and ClusterGroup classes
"""
from jlgh import DIM
from jlgh.lgh import parse_spec_coord

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
        self.mult = len(self.species_coords)

    def __repr__(self):
        rep = '('
        for spec, coord in self.species_coords:
            rep += "'{0}@{1}.({2},{3},0)',".format(spec,
                                             coord.name,
                                             coord.offset[0],
                                             coord.offset[1])
        rep += ')'
        return rep

class ClusterGroup(object):
    """
    Class that holds a cluster group. That is all clusters that are equivalent through
    symmetry transformations
    """
    def __init__(self,
                 name,
                 energy,
                 cluster_list,
                 enabled = True,
                 fixed = False):
        self.name = name

        self.eini = energy
        self.e = energy

        self.clusters = []
        for cluster_def in cluster_list:
            if isinstance(cluster_def,Cluster):
                self.clusters.append(cluster_def)
            else:
                self.clusters.append(Cluster(cluster_def))
        self.nclusters = len(self.clusters)
        mult = list(set([x.mult for x in self.clusters]))
        if len(mult) > 1:
            raise ValueError('ClusterGroup can only contain clusters '
                             'with the same multiplcity')
        self.mult = mult[0]
        self.enabled = enabled
        self.fixed = fixed

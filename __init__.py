"""
Module used to convert lattice definitions to
the objects used to define the LGH
"""
import ase
import ase.io
from ase.constraints import FixAtoms, FixedLine
import ase.lattice.surface
import numpy as np
from struformat.molcrys import cluster
import time, scipy, sys, os, glob

DIM = 2  # We use this as a parameter, but probably will always be fixed
DELTA_H_SMALL = 1.0 # Default height over the surface to place adsorbates

TOL_ANGLE = 1e-9
TOL_DIST = 1e-5

zvect = np.array([0.,0.,1.])

from jlgh.lgh import LGH, BaseCell, Adsorbate, Site, BareCoord
from jlgh.cluster import Cluster, ClusterGroup
from jlgh.config import Config

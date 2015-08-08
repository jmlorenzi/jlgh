from jlgh import *
import numpy as np
import ase.lattice.surface as surf
from ase.io.castep import read_cell
from ase.visualize import view
from ase.constraints import FixAtoms
from ase.utils.geometry import get_layers

atoms = read_cell('Pd_O_O_NO_CO_0000.cell')
atoms.calc = None

Pd_latconst = 3.937211

slab = surf.fcc100('Pd',a = Pd_latconst,size = [1,1,4],vacuum = 10)
slab.set_constraint(FixAtoms(indices = [ il[0] for il
                        in enumerate(get_layers(slab,(0,0,1))[0])
                        if il[1] < 2]))

h0 = max([at.position[2] for at in slab]) / slab.cell[2][2]

sites = [ Site(name='hollow', pos=[.5,.5,h0]),
          Site(name='top', pos=[.0,.0,h0]),
          Site(name='br1', pos=[.5,.0,h0]),
          Site(name='br2', pos=[.0,.5,h0]),
         ]

ucell = SurfUnitCell(slab,sites_list=sites)

COatoms = ase.Atoms('CO',positions=[[0.0,0.0,1.0],[0.0,0.0,2.0]])
NOatoms = ase.Atoms('NO',positions=[[0.0,0.0,1.0],[0.0,0.0,2.0]])
NO2atoms = ase.Atoms('NO2',positions=[[0.0,0.0,1.0],[-0.5,0.0,1.5],[0.5,0.0,1.5]],)
Oatoms = ase.Atoms('O',positions=[[0.0,0.0,1.0],])

COads = Adsorbate('CO',COatoms,center=[0.,0.,0.])
NOads = Adsorbate('NO',NOatoms,center=[0.,0.,0.])
NO2ads = Adsorbate('NO2',NO2atoms,center=[0.,0.,0.])
Oads = Adsorbate('O',Oatoms,center=[0.,0.,0.])

lgh = LGH(sucell = ucell)
lgh.add_adsorbates([COads,Oads,NOads,NO2ads])
lgh.read_atoms(atoms)

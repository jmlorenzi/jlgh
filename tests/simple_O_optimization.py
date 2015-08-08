from jlgh import *
import numpy as np
import ase.lattice.surface as surf
from ase.constraints import FixAtoms
from ase.utils.geometry import get_layers

Pd_latconst = 3.937211

slab = surf.fcc100('Pd',a = Pd_latconst,size = [1,1,4],vacuum = 10)
slab.set_constraint(FixAtoms(indices = [ il[0] for il
                        in enumerate(get_layers(slab,(0,0,1))[0])
                        if il[1] < 2]))

h0 = max([at.position[2] for at in slab]) / slab.cell[2][2]

sites = [ Site(name='hollow', pos=[.5,.5,h0]),]

E_2x2 = -56199.75241886

base_cell = BaseCell(slab,sites_list=sites,energy=E_2x2/4.)


Oatoms = ase.Atoms('O',positions=[[0.0,0.0,1.0],])
Oads = Adsorbate('O',Oatoms,center=[0.,0.,0.], binding_energy = -439.4812229)

V_O_O_1NN_defs = [ ('O@hollow.(0,0,0)','O@hollow.(1,0,0)'),
                   ('O@hollow.(0,0,0)','O@hollow.(0,1,0)'),]
V_O_O_1NN = ClusterGroup('V_O_O_1NN',0.3,V_O_O_1NN_defs)

V_O_O_2NN_defs = [ ('O@hollow.(0,0,0)','O@hollow.(1,1,0)'),
                   ('O@hollow.(0,0,0)','O@hollow.(1,-1,0)'),]
V_O_O_2NN = ClusterGroup('V_O_O_2NN',0.07,V_O_O_2NN_defs)

enes_defs = [ (-57077.82959288, ('O@hollow.(0,0,0)','O@hollow.(1,0,0)')),
              (-57078.30630942, ('O@hollow.(0,0,0)','O@hollow.(1,1,0)')),
              (-57516.23747937, ('O@hollow.(0,0,0)','O@hollow.(1,0,0)','O@hollow.(1,1,0)')),
              (-57953.93299920, ('O@hollow.(0,0,0)','O@hollow.(1,0,0)','O@hollow.(0,1,0)','O@hollow.(1,1,0)')),
              ]

conf_list = [Config(size=[2,2],species_coords = ed[1],energy =ed [0]) for ed in enes_defs]

lgh = LGH(base_cell = base_cell)
lgh.add_adsorbate(Oads)
lgh.add_clustergroups([V_O_O_1NN,V_O_O_2NN])
lgh.add_configs(conf_list)
lgh.reset()

optim = LGHOptimizer(lgh)

optim.run(verbose=2)

print(lgh.base_energy)
print(lgh.binding_energies)
print(lgh.cluster_energies)

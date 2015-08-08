import ase
import ase.lattice.surface as surf
from ase.visualize import view
import jlgh as jl

slab = surf.fcc100('Pd',[1,1,4],vacuum = 10)
# view(slab*[2,2,1])
# raise SystemExit()

h0 = max([at.position[2] for at in slab]) / slab.cell[2][2]

sites = [ jl.Site(name='hollow', pos=[.5,.5,h0]),
          jl.Site(name='top', pos=[.0,.0,h0]),
          jl.Site(name='br1', pos=[.5,.0,h0]),
          jl.Site(name='br2', pos=[.0,.5,h0]),
            ]

ucell = jl.SurfUnitCell(slab,sites_list=sites)

COatoms = ase.Atoms('CO',positions=[[0.0,0.0,1.0],[0.0,0.0,2.0]])
Oatoms = ase.Atoms('O',positions=[[0.0,0.0,1.0],])

COads = jl.Adsorbate('CO',COatoms,center=[0.,0.,0.])
Oads = jl.Adsorbate('O',Oatoms,center=[0.,0.,0.])

# species_coords = [ 'O@hollow.(0,0,0)', 'O@hollow.(1,0,0)']
# species_coords = [ 'O@hollow.(0,0,0)', 'O@hollow.(1,0,0)','O@hollow.(1,1,0)']
species_coords = [ 'O@hollow.(0,0,0)', 'O@hollow.(1,0,0)','O@hollow.(1,1,0)','O@hollow.(0,1,0)']
species_coords = [ 'O@hollow.(0,0,0)', 'CO@br2.(1,1,0)']
species_coords = [ 'O@hollow.(0,0,0)', 'O@hollow.(1,0,0)','CO@br2.(1,1,0)']
conf = jl.Config(size = [2,2],species_coords = species_coords)

V_O_CO_defs = [  ('O@hollow.(0,0,0)','CO@br2.(1,1,0)'),
                 ('O@hollow.(0,0,0)','CO@br2.(0,1,0)'),
                 ('O@hollow.(0,0,0)','CO@br1.(-1,1,0)'),
                 ('O@hollow.(0,0,0)','CO@br1.(-1,0,0)'),
                 ('O@hollow.(0,0,0)','CO@br2.(0,-1,0)'),
                 ('O@hollow.(0,0,0)','CO@br2.(1,-1,0)'),
                 ('O@hollow.(0,0,0)','CO@br1.(1,0,0)'),
                 ('O@hollow.(0,0,0)','CO@br1.(1,1,0)'),]

V_O_CO = jl.ClusterGroup('V_O_CO',0.0,V_O_CO_defs)


V_O_O_1NN_defs = [ ('O@hollow.(0,0,0)','O@hollow.(1,0,0)'),
                   ('O@hollow.(0,0,0)','O@hollow.(0,1,0)'),]

V_O_O_1NN = jl.ClusterGroup('V_O_O_1NN',0.0,V_O_O_1NN_defs)


# cg2 = jl.ClusterGroup('V_O_2NN',0.0,cluster_list2)


# species_coords = [ ('O',jl.BareCoord('hollow',(0,0))),
#                    ('CO',jl.BareCoord('br1',(1,1))),
#                    ]
# config2 = jl.Config(size = [2,2],species_coords = species_coords)

lgh = jl.LGH(sucell = ucell)
lgh.add_adsorbates([COads,Oads])
lgh.add_configs([conf,])
lgh.add_clustergroups([V_O_O_1NN,V_O_CO])

for config in lgh.config_list:
    conf_atoms = config.return_atoms()
    view(conf_atoms)
    config.calculate_matrix()
    # print('Config matrix =')
    # print(config.matrix)
    # print(' ')

    config.identify_clusters()

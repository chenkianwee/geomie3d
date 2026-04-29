import numpy as np
import geomie3d
import geomie3d.viz

box = geomie3d.create.box(1, 1, 1)
mv_box = geomie3d.modify.move_topo(box, [.5,.5,0], np.array([0,0,0]))

rot_box = geomie3d.modify.rotate_topo(box, [0,1,0], -60)

geomie3d.viz.viz([{'topo_list':[rot_box], 'colour': 'white', 'draw_edges':False},
                  {'topo_list':[mv_box], 'colour': 'red', 'draw_edges':False}])

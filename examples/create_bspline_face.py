import geomie3d
import geomie3d.viz

#using two edges and lofting them
ctrlpts1 = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [1,-1,0], [0, 0, 0]]
e1 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts1, degree=2, resolution=0.01, 
                                                attributes = {})

ctrlpts2 = [[0, 0, 2], [1, 1, 2], [2, 0, 2], [1,-1,2], [0, 0, 2]]
e2 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts2, degree=2, resolution=0.01, 
                                           attributes = {})

elist = [e1,e2]
f = geomie3d.create.bspline_face_frm_loft(elist)
g = geomie3d.create.grids_frm_bspline_face(f, 4, 4)
print('number of grids', len(g))
v = geomie3d.create.grid_pts_frm_bspline_face(f, 15, 15)
# geomie3d.viz.viz([{'topo_list': [f], 'colour': 'blue'},
#                   {'topo_list': v, 'colour': 'red'}])
geomie3d.viz.viz([{'topo_list': [f], 'colour': 'blue'}])

# using control points
ctrl_pts = [[-25,-25,-5],[-25,-15,0],[-25,-5,0],[-25,5,0],[-25,15,0],[-25,25,5], 
            [-15,-25,0], [-15,-15,0],[-15,-5,0],[-15,5,0],[-15,15,1],[-15,25,1], 
            [-5,-25,5],[-5,-15,5],[-5,-5,5],[-5,5,0],[-5,15,0],[-5,25,0], 
            [5,-25,0],[5,-15,0],[5,-5,0],[5,5,-5],[5,15,-5],[5,25,-5], 
            [15,-25,0],[15,-15,0],[15,-5,0],[15,5,0],[15,15,-10],[15,25,-10],
            [25,-25,0], [25,-15,0], [25,-5,0],[25,5,0],[25,15,0],[25,25,-5]]

# ctrl_pts = [[-25,-25,0], [-25,-15,0], [-25,-5,0], [-25,5,0], [-25,15,0], [-25,25,0], 
#             [-15,-25,0], [-15,-15,0], [-15,-5,0], [-15,5,0], [-15,15,0], [-15,25,0], 
#             [-5,-25,0], [-5,-15,0], [-5,-5,0], [-5,5,0], [-5,15,0], [-5,25,0], 
#             [5,-25,0], [5,-15,0], [5,-5,0], [5,5,-0], [5,15,0], [5,25,0], 
#             [15,-25,0], [15,-15,0], [15,-5,0], [15,5,0], [15,15,0], [15,25,0],
#             [25,-25,0], [25,-15,0], [25,-5,0], [25,5,0], [25,15,0], [25,25,0]]

# ctrl_pts = [[1,5,0], [5,5,0],
#             [1,0,0], [5,0,0]]

deg_u = 1
deg_v = 1

kv_u = 6
kv_v = 6

f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v, 
                                             resolution=0.167)
vs1 = geomie3d.create.vertex_list(ctrl_pts)
surf_pts = f.surface.evalpts
vs2 = geomie3d.create.vertex_list(surf_pts)

geomie3d.viz.viz([{'topo_list': [f], 'colour': 'blue'},
                  {'topo_list':vs1, 'colour': 'red'},
                  {'topo_list':vs2, 'colour': 'blue'}])

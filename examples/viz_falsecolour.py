import geomie3d
import geomie3d.viz

ctrl_pts = [[1,5,0], [5,5,0],
            [1,0,0], [5,0,0]]

deg_u = 1
deg_v = 1

kv_u = 2
kv_v = 2

f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v)
g = geomie3d.create.grids_frm_bspline_face(f, 5, 5)

res = []
for i in range(len(g)):
    res.append(i*10)

ctrlpts1 = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [1,-1,0], [0, 0, 0]]
e1 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts1, degree=2, resolution=0.01, 
                                           attributes = {})


ctrlpts2 = [[0, 0, 2], [1, 1, 2], [2, 0, 2], [1,-1,2], [0, 0, 2]]
e2 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts2, degree=2, resolution=0.01, 
                                           attributes = {})

elist = [e1,e2]
f2 = geomie3d.create.bspline_face_frm_loft(elist)
v = geomie3d.create.grid_pts_frm_bspline_face(f2, 15, 15)


geomie3d.viz.viz_falsecolour(g, res, false_min_max_val=[30,120],
                                  other_topo_dlist = [{'topo_list': [f2], 'colour': 'blue'},
                                                    {'topo_list': v, 'colour': 'red'}])
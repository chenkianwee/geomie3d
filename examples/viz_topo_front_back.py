import geomie3d
import geomie3d.viz

# the geometry data to view
ctrl_pts = [[1,5,0], [5,5,0],
            [1,0,0], [5,0,0]]

deg_u = 1
deg_v = 1

kv_u = 2
kv_v = 2

f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v)
g = geomie3d.create.grids_frm_bspline_face(f, 5, 5)

bx = geomie3d.create.box(5, 5, 5)
edges1 = geomie3d.get.edges_frm_solid(bx)

topo_ds = [{'topo_list': [bx], 'colour': [1,0,0,1], 'draw_edges' : False}, {'topo_list': g, 'colour': 'green', 'draw_edges': False}]
geomie3d.viz.viz(topo_ds, gl_option = 'opaque')
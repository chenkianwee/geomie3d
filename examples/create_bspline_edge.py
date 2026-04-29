import geomie3d
import geomie3d.viz

ctrl_pts = [[0,0,0], [0,20,0], [10,0,0], [20,0,0]]
e = geomie3d.create.bspline_edge_frm_xyzs(ctrl_pts, degree=2, resolution=0.01)
vs3 = geomie3d.create.vertex_list(ctrl_pts)
edge1 = geomie3d.create.pline_edge_frm_verts(vs3)

# Get curve points
points = e.curve.evaluate_single(0.5)
# print(points)
points = e.curve.evalpts
# print(points)
vs = geomie3d.create.vertex_list(points)
edge = geomie3d.create.pline_edge_frm_verts(vs)
vs2 = geomie3d.get.vertices_frm_edge(edge)

geomie3d.viz.viz([{'topo_list':[edge], 'colour':'red'}])

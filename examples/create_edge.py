import geomie3d
import geomie3d.viz

#create polyline edge
xyz_list1 = [(10,10,0), (20,10,0), (20,20,0)]
att_list1 = [{"type":"bus stop"}, {"type":"shop"}, {"type":"monument"}]
vlist1 = geomie3d.create.vertex_list(xyz_list1, attributes_list = att_list1)
pl_att1 = {"id":0, "type": "sidewalk"}
pline_edge1 = geomie3d.create.pline_edge_frm_verts(vlist1, attributes = pl_att1)

xyz_list2 = [(20,20,0), (10,20,0), (10,10,0)]
att_list2 = [{"type":"bus stop"}, {"type":"shop"}, {"type":"bus stop", "id":100}]
vlist2 = geomie3d.create.vertex_list(xyz_list2, attributes_list = att_list2)
pl_att2 = {"id":1, "type": "road"}
pline_edge2 = geomie3d.create.pline_edge_frm_verts(vlist2, attributes = pl_att2)

print(pline_edge1.__dict__, pline_edge2.curve)

#create bspline edge
ctrlpts = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [1,-1,0], [0,0,0]]
bspline_edge = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts, degree=2, resolution=0.01)
vs = geomie3d.get.vertices_frm_edge(bspline_edge)

f = geomie3d.create.polygon_face_frm_verts(vs)
# geomie3d.viz.viz([{'topo_list': [bspline_edge], 'colour': 'blue'}])

geomie3d.viz.viz([{'topo_list': [pline_edge1], 'colour': 'blue'}, 
                  {'topo_list': [bspline_edge], 'colour': 'red'}])
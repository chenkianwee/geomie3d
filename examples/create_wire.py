import geomie3d
import geomie3d.viz

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

edge_list = [pline_edge1, pline_edge2]
w_att = {"id":0, "type": "plot"}
wire = geomie3d.create.wire_frm_edges(edge_list, attributes = w_att)
geomie3d.viz.viz([{'topo_list': [wire], 'colour': 'red'}])
print(wire.attributes)
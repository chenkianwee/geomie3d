import geomie3d
import geomie3d.viz

#defining the boundary wire
xyz_list1 = [(10,10,0), (20,10,0), (20,20,0)]
att_list1 = [{"type":"bus stop"}, {"type":"shop"}, {"type":"monument"}]
vlist1 = geomie3d.create.vertex_list(xyz_list1, attributes_list = att_list1)
pl_att1 = {"id":0, "type": "sidewalk"}
pline_edge1 = geomie3d.create.pline_edge_frm_verts(vlist1, attributes = pl_att1)

xyz_list2 = [(20,20,0), (10,20,0), (10,10,0)]
att_list2 = [{"type":{"test":1}}, {"type":"shop"}, {"type":"bus stop", "id":100}]
vlist2 = geomie3d.create.vertex_list(xyz_list2, attributes_list = att_list2)
pl_att2 = {"id":1, "type": "road"}
pline_edge2 = geomie3d.create.pline_edge_frm_verts(vlist2, attributes = pl_att2)

bdry_edge_list = [pline_edge1, pline_edge2]
w_att1 = {"id":0, "type": "plot"}
bdry_wire = geomie3d.create.wire_frm_edges(bdry_edge_list, attributes = w_att1)

#defining the hole wire
xyz_list3 = [(12,12,0), (18,12,0), (18,18,0)]
vlist3 = geomie3d.create.vertex_list(xyz_list3)
pline_edge3 = geomie3d.create.pline_edge_frm_verts(vlist3)

xyz_list4 = [(18,18,0), (12,18,0), (12,12,0)]
vlist4 = geomie3d.create.vertex_list(xyz_list4)
pline_edge4 = geomie3d.create.pline_edge_frm_verts(vlist4)

hole_edge_list = [pline_edge3, pline_edge4]
hole_wire = geomie3d.create.wire_frm_edges(hole_edge_list)

#create the face
f_att = {"type": "residential", "id": 0}
face = geomie3d.create.polygon_face_frm_wires(bdry_wire, hole_wire_list = [hole_wire], 
                                              attributes = f_att)
xyz_list = [[20,0,0], 
            [20,5,0],
            [10,5,0],
            [10,10,0],
            [0,10,0],
            [0,0,0]]

vlist = geomie3d.create.vertex_list(xyz_list)
face2 = geomie3d.create.polygon_face_frm_verts(vlist)
tri_faces = geomie3d.modify.triangulate_face(face2)
tri_faces2 = geomie3d.modify.triangulate_face(face2, indices=True)
print(tri_faces2)
for tri_f in tri_faces[0:1]:
    vs = geomie3d.get.vertices_frm_face(tri_f)

geomie3d.viz.viz([{'topo_list':tri_faces, 'colour': 'red'}])
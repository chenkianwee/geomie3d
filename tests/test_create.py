import geomie3d

def test_create_box():
    box = geomie3d.create.box(10, 5, 1)
    shell = box.shell
    face_list = shell.face_list
    # geomie3d.utility.viz([{'topo_list': face_list, 'colour': 'BLUE'}])
    
    for face in face_list:
        wire = face.bdry_wire
        vlist = geomie3d.get.points_frm_wire(wire)
        tri_face = geomie3d.modify.triangulate_face(face)
        
    print(tri_face, vlist)
    
def test_create_edge():
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
    print(bspline_edge)
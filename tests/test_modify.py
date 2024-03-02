import geomie3d 
import numpy as np

def test_modify_rev_face_normal():
    #reverse polygon face
    bxyzs = [[5,-5, 0], [5, 5, 0], [-5, 5, 0], [-5, -5, 0]]

    #holes need to be the reverse of the boundary
    hxyzs = [[[2.5, -2.5, 0], [-2.5, -2.5, 0], [-2.5, 2.5, 0], [2.5, 2.5, 0]]]

    bverts = geomie3d.create.vertex_list(bxyzs)
    hvert_ls = [geomie3d.create.vertex_list(h) for h in hxyzs]
    f = geomie3d.create.polygon_face_frm_verts(bverts, hole_vertex_list = hvert_ls)

    flipped_f = geomie3d.modify.reverse_face_normal(f)

    n1 = geomie3d.get.face_normal(f)
    n2 = geomie3d.get.face_normal(flipped_f)

    #reverse bspline face
    #using two edges and lofting them
    ctrlpts1 = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [1,-1,0], [0, 0, 0]]
    e1 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts1, degree=2, resolution=0.01, attributes = {})
    ctrlpts2 = [[0, 0, 2], [1, 1, 2], [2, 0, 2], [1,-1,2], [0, 0, 2]]
    e2 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts2, degree=2, resolution=0.01, attributes = {})
    elist = [e1,e2]
    f = geomie3d.create.bspline_face_frm_loft(elist)
    flipped_f = geomie3d.modify.reverse_face_normal(f)

    n1 = geomie3d.get.face_normal(f)
    n2 = geomie3d.get.face_normal(flipped_f)

def test_modify_rotate():
    box = geomie3d.create.box(1, 1, 1)
    mv_box = geomie3d.modify.move_topo(box, [.5,.5,0], np.array([0,0,0]))
    rot_box = geomie3d.modify.rotate_topo(box, [0,1,0], -60)

def test_modify_triangulate_bspline_face():
    ctrl_pts = [[-25,-25,-5],[-25,-15,0],[-25,-5,0],[-25,5,0],[-25,15,0],[-25,25,5], 
                [-15,-25,0], [-15,-15,0],[-15,-5,0],[-15,5,0],[-15,15,1],[-15,25,1], 
                [-5,-25,5],[-5,-15,5],[-5,-5,5],[-5,5,0],[-5,15,0],[-5,25,0], 
                [5,-25,0],[5,-15,0],[5,-5,0],[5,5,-5],[5,15,-5],[5,25,-5], 
                [15,-25,0],[15,-15,0],[15,-5,0],[15,5,0],[15,15,-10],[15,25,-10],
                [25,-25,0], [25,-15,0], [25,-5,0],[25,5,0],[25,15,0],[25,25,-5]]
    deg_u = 2
    deg_v = 2
    kv_u = 6
    kv_v = 6
    f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v, 
                                             resolution=0.167)
    n = geomie3d.get.face_normal(f)
    tri = geomie3d.modify.triangulate_face(f)
    grids = geomie3d.create.grids_frm_bspline_face(f, 4, 5)

def test_modify_triangulate_face():
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
    for tri_f in tri_faces[0:1]:
        vs = geomie3d.get.vertices_frm_face(tri_f)

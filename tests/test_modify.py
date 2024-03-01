import geomie3d

def test_modify_rev_face_normal():
    #========================================================================
    #reverse polygon face
    #========================================================================
    bxyzs = [[5,-5, 0],
            [5, 5, 0],
            [-5, 5, 0],
            [-5, -5, 0]]

    #holes need to be the reverse of the boundary
    hxyzs = [[[2.5, -2.5, 0],
            [-2.5, -2.5, 0],
            [-2.5, 2.5, 0],
            [2.5, 2.5, 0]]]

    bverts = geomie3d.create.vertex_list(bxyzs)
    hvert_ls = [geomie3d.create.vertex_list(h) for h in hxyzs]
    f = geomie3d.create.polygon_face_frm_verts(bverts, 
                                                hole_vertex_list = hvert_ls)

    flipped_f = geomie3d.modify.reverse_face_normal(f)

    n1 = geomie3d.get.face_normal(f)
    n2 = geomie3d.get.face_normal(flipped_f)

    #========================================================================
    #reverse bspline face
    #========================================================================
    #using two edges and lofting them
    ctrlpts1 = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [1,-1,0], [0, 0, 0]]
    e1 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts1, degree=2, resolution=0.01, 
                                            attributes = {})
    ctrlpts2 = [[0, 0, 2], [1, 1, 2], [2, 0, 2], [1,-1,2], [0, 0, 2]]
    e2 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts2, degree=2, resolution=0.01, 
                                            attributes = {})
    elist = [e1,e2]
    f = geomie3d.create.bspline_face_frm_loft(elist)
    flipped_f = geomie3d.modify.reverse_face_normal(f)

    n1 = geomie3d.get.face_normal(f)
    n2 = geomie3d.get.face_normal(flipped_f)
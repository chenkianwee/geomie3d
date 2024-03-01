import geomie3d
import numpy as np

def test_are_bbox1_related_bbox2():
    bbox1 = geomie3d.utility.Bbox([1,1,0,10,10,5])
    bbox2 = geomie3d.utility.Bbox([1,1,0,10,10,5])

    bbox3 = geomie3d.utility.Bbox([2,2,1,8,8,4])
    bbox4 = geomie3d.utility.Bbox([2,2,1,8,8,4])

    bbox5 = geomie3d.utility.Bbox([12,12,11,18,18,14])
    bbox6 = geomie3d.utility.Bbox([15,19,11,21,23,14])

    are_related = geomie3d.calculate.are_bboxes1_related2_bboxes2([bbox1, bbox2, bbox5], [bbox3, bbox4, bbox6])

def test_bboxes_center():
    bbox1 = geomie3d.create.bbox_frm_arr([1,1,0,10,10,5])
    bbox2 = geomie3d.create.bbox_frm_arr([2,2,1,8,8,4])
    center_pts = geomie3d.calculate.bboxes_centre([bbox1, bbox2])

def test_calc_cs2cs_mat():
    pass

def test_calc_dist_btw_xyzs():
    xyz1 = [[0,0,0], [1,1,2], [3,3,3]]
    xyz2 = [0,0,0]
    dist = geomie3d.calculate.dist_btw_xyzs(xyz1, xyz2)
    # print(dist)

    xyz1 = np.array([2.23912, -19.36, 0])
    xyz2 = np.array([11.86, -18.15, 0])
    vec1 = xyz2 - xyz1
    norm = geomie3d.calculate.normalise_vectors([vec1])

def test_calc_distance_btw_vertices2line_edges():
    pointxyzs = [[1,6,1], [8,2,0]]
    linexyzs = [[[1,6,0], [5,6,0]],
                [[1,6,0], [5,6,2]]]

    vlist = geomie3d.create.vertex_list(pointxyzs)

    edge_ls = []
    for linexyz in linexyzs:
        verts = geomie3d.create.vertex_list(linexyz)
        edge = geomie3d.create.pline_edge_frm_verts(verts)
        edge_ls.append(edge)
        
    dists, int_vs = geomie3d.calculate.dist_vertex2line_edge(vlist, edge_ls, int_pts = True)

def test_calc_face_area():
    poly_xyzs = [[0, 0, 0], [10, 0, 0], [10, 3, 4], [0, 3, 4]]
    poly_xyzs = [[0,10,0], [0,0,0], [5,0,0], [5,2,0], [3,2,0], [3,10,0]]

    vs = geomie3d.create.vertex_list(poly_xyzs)
    face = geomie3d.create.polygon_face_frm_verts(vs)

    area = geomie3d.calculate.face_area(face)

def test_calc_find_connected_edges():
    xyz_list = [[[2,3,0], [5,3,0]],
            [[6,9,0], [4,9,1]],
            [[5,8,0], [2,8,0]],
            [[5,3,0], [5,8,0]],
            [[2,8,0], [2,3,0]],
            [[5,3,0], [6,1,0]],
            [[5,8,0], [6,9,0]],
            ]

    edge_list = []
    for cnt,xyzs in enumerate(xyz_list):
        vertices = geomie3d.create.vertex_list(xyzs)
        e = geomie3d.create.pline_edge_frm_verts(vertices, attributes = {'id': cnt})
        edge_list.append(e)

    connected_indxs = geomie3d.calculate.a_connected_path_from_edges(edge_list, indx=True)
    connected = geomie3d.calculate.a_connected_path_from_edges(edge_list, indx=False)

def test_calc_intersection_2faces():
    pass

def calc_intersection2lines():
    linexyzs1 = [[[1,2,0],[6,5,0]],
             [[1,1,0],[1,6,0]],
             [[3,2,0],[6,5,0]]]

    linexyzs2 = [[[6,2,0],[1,5,0]],
                [[6,2,0],[1,5,0]],
                [[6,1,0],[6,6,0]],
                ]
    
    def line2edge(linexyzs):
        edge_ls = []
        for linexyz in linexyzs:
            vlist = geomie3d.create.vertex_list(linexyz)
            edge = geomie3d.create.pline_edge_frm_verts(vlist)
            edge_ls.append(edge)
        return edge_ls

    edge_list1 = line2edge(linexyzs1)
    edge_list2 = line2edge(linexyzs2)

    int_vs = geomie3d.calculate.lineedge_intersect(edge_list1, edge_list2)

def test_is_anticlockwise():
    xyzs = [[2,6,0], [2,1,0], [3,3,0]]
    xyzs = [[2,6,0], [2,1,0], [2,5,0], [2,9,0]]

    xyzs = [[[2,6,0], [2,1,0], [3,3,0]],
            [[2,6,0], [2,1,0], [1,3,0]],
            [[2,6,0], [2,1,0], [2,3,0]]]

    ref_vec = [0,0,1]
    ref_vec = [[0,0,1],
            [0,0,1],
            [0,0,1]]

    is_anti = geomie3d.calculate.is_anticlockwise(xyzs, ref_vec)

def test_calc_is_coplanar_collinear():
    xyzs = [[[0,0,0], [3,0,0],[2,3,0],[0,3,1]],
        [[0,0,0], [3,0,0],[2,3,0],[1,3,0]]]

    vs2 = [geomie3d.create.vertex_list(xyz) for xyz in xyzs]

    xyzs = [[0,0,0], [3,0,0], [2,3,0], [0,3,0], [0,3,0], [0,0,0]]
    is_coplanar = geomie3d.calculate.is_coplanar_xyzs(xyzs)
    is_coplanar = geomie3d.calculate.is_coplanar(vs2)

    a = np.array(xyzs)
    centre_pt = geomie3d.calculate.xyzs_mean(xyzs)
    print(centre_pt)

    xyzs2 = [[0,0,0],
            [0,10,0],
            [0,20,1]]

    vs2 = geomie3d.create.vertex_list(xyzs2)
    is_collinear = geomie3d.calculate.is_collinear(vs2)

def test_calc_linexyzs_frm_ts():
    linexyzs = [[[1,0,0], [1,6,0]], 
                [[2,2,0], [2,8,0]]]
    ts = [1.5, 0.5]

    xyzs = geomie3d.calculate.linexyzs_from_t(ts, linexyzs)

def test_calc_non_dup_edges():
    box = geomie3d.create.box(10, 10, 10)
    box = geomie3d.modify.move_topo(box, (0,0,5), (0,0,0))
    wires = geomie3d.get.wires_frm_solid(box)
    # geomie3d.viz.viz([{'topo_list':wires, 'colour':'red'}])
    face_list = geomie3d.get.faces_frm_solid(box)

    tri_faces_ls = []
    for cnt,f in enumerate(face_list):
        tri_faces = geomie3d.modify.triangulate_face(f, indices=False)
        tri_faces_ls.extend(tri_faces)

    f1 = geomie3d.create.polygon_face_frm_midpt([0,0,0], 5,5,5)
    f1 = geomie3d.modify.rotate_topo(f1, [1,0,0], 45.0)

    tri_faces_ls.append(f1)
    grp_faces, indv_faces = geomie3d.calculate.grp_faces_on_nrml(tri_faces_ls, return_idx=False)
    outline_ls = []
    for grp  in grp_faces:
        outline_edges, dup_edges = geomie3d.calculate.find_faces_outline(grp)
        print(dup_edges)
        # {'topo_list':grp, 'colour': 'red'},
        outline_ls.extend(outline_edges)

    outline2, dup2 = geomie3d.calculate.find_non_dup_lineedges(outline_ls)

def test_calc_rays_bboxes_intersect():
    rays_xyz = [[[1, 0, 0], [0,0,1]],
                [[4, 3, 10], [1,0,0]]]
    bbox_arr_ls = [[5, -1, 5, 8, 3, 10],
                    [-2, -2, 4, 2, 2, 12]]

    #=============================================================================
    def bbox2box(bbox):
        dimx = bbox.maxx - bbox.minx
        dimy = bbox.maxy - bbox.miny
        dimz = bbox.maxz - bbox.minz
        centre_pt = [dimx/2+bbox.minx, dimy/2+bbox.miny, dimz/2+bbox.minz]
        bx = geomie3d.create.box(dimx, dimy, dimz, centre_pt = centre_pt)
        return bx
    #=============================================================================

    ray_ls = [geomie3d.create.ray(ray_xyz[0], ray_xyz[1]) for ray_xyz in rays_xyz]
    bbox_list = [geomie3d.create.bbox_frm_arr(bbox_arr) for bbox_arr in bbox_arr_ls]

    inter_res = geomie3d.calculate.rays_bboxes_intersect(ray_ls, bbox_list)
    hrs = inter_res[0]
    mrs = inter_res[1]
    hbs = inter_res[2]
    mbs = inter_res[3]

    #for viz
    dir_es = []
    if len(hrs) != 0:
        invs = []
        intes = []
        for hr in hrs:
            o = hr.origin
            d = hr.dirx
            dir_mv = o + d*10
            v_dir = geomie3d.create.vertex(dir_mv)
            v = geomie3d.create.vertex(o)
            invs.append(v)
            dir_e = geomie3d.create.pline_edge_frm_verts([v, v_dir])
            dir_es.append(dir_e)
            
            att = hr.attributes['rays_bboxes_intersection']
            intersects = att['intersection']
            for inter in intersects:
                inv = geomie3d.create.vertex(inter)
                invs.append(inv)
                inte = geomie3d.create.pline_edge_frm_verts([v, inv])
                intes.append(inte)
                
    if len(mrs) != 0:
        vs = []
        for mr in mrs:
            o = mr.origin
            d = mr.dirx
            dir_mv = o + d*10
            v_dir = geomie3d.create.vertex(dir_mv)
            v = geomie3d.create.vertex(o)
            vs.append(v)
            dir_e = geomie3d.create.pline_edge_frm_verts([v, v_dir])
            dir_es.append(dir_e)

    if len(hbs) != 0:
        hbx = []
        for hb in hbs:
            bx = bbox2box(hb)
            bx = geomie3d.get.edges_frm_solid(bx)
            hbx.extend(bx)

    if len(mbs) != 0:
        mbx = []
        for mb in mbs:
            bx = bbox2box(mb)
            bx = geomie3d.get.edges_frm_solid(bx)
            mbx.extend(bx)

def test_calc_rays_bboxes_intersect2():
    v_size = 10
    ray_orig = [0,0,0]
    #-----------------------------------------------------------------------------------------
    def convert_polygon2bspline_face(poly_face):
        verts = geomie3d.get.vertices_frm_face(poly_face)
        pts = np.array([v.point.xyz for v in verts])
        pts = np.array([pts[2], pts[1], pts[3], pts[0]])
        bface = geomie3d.create.bspline_face_frm_ctrlpts(pts, 2, 2, 1, 1)
        return bface

    def bbox2box(bbox):
        dimx = bbox.maxx - bbox.minx
        dimy = bbox.maxy - bbox.miny
        dimz = bbox.maxz - bbox.minz
        centre_pt = [dimx/2+bbox.minx, dimy/2+bbox.miny, dimz/2+bbox.minz]
        bx = geomie3d.create.box(dimx, dimy, dimz, centre_pt = centre_pt)
        return bx

    bx = geomie3d.create.box(20,20,20)
    bfaces = geomie3d.get.faces_frm_solid(bx)
    bbx_ls = []
    e_ls = []
    bx_ls = []
    for f in bfaces:
        bspline_f = convert_polygon2bspline_face(f)
        gfs = geomie3d.create.grids_frm_bspline_face(bspline_f, 1, 1)
        for gf in gfs:
            midpt = geomie3d.calculate.face_midxyz(gf)
            bbx = geomie3d.create.bboxes_frm_midpts([midpt], [v_size], [v_size], [v_size])[0]
            bbx_ls.append(bbx)
            bx = geomie3d.create.box(v_size, v_size, v_size, centre_pt = midpt)
            bx_ls .append(bx)
            bedges = geomie3d.get.edges_frm_solid(bx)
            e_ls.extend(bedges)

    ndir = 360
    unitball = geomie3d.d4pispace.tgDirs(ndir)
    #create the rays for each analyse pts
    aly_vs = []
    rays = []
    v_ls = []
    for dix in unitball.getDirList():
        dirx = [dix.x, dix.y, dix.z]
        vertex = geomie3d.create.vertex(dirx)
        v_ls.append(vertex)
        ray = geomie3d.create.ray(ray_orig, dirx)
        rays.append(ray)

    # print('dirx', rays[6].dirx)
    inter_res = geomie3d.calculate.rays_bboxes_intersect(rays, bbx_ls)

    hrs = inter_res[0]
    mrs = inter_res[1]
    hbs = inter_res[2]
    mbs = inter_res[3]

    #for viz
    dir_es = []
    if len(hrs) != 0:
        invs = []
        intes = []
        for hr in hrs:
            o = hr.origin
            d = hr.dirx
            dir_mv = o + d*10
            v_dir = geomie3d.create.vertex(dir_mv)
            v = geomie3d.create.vertex(o)
            invs.append(v)
            dir_e = geomie3d.create.pline_edge_frm_verts([v, v_dir])
            dir_es.append(dir_e)
            
            att = hr.attributes['rays_bboxes_intersection']
            intersects = att['intersection']
            for inter in intersects:
                inv = geomie3d.create.vertex(inter)
                invs.append(inv)
                inte = geomie3d.create.pline_edge_frm_verts([v, inv])
                intes.append(inte)
        
    if len(mrs) != 0:
        vs = []
        for mr in mrs:
            o = mr.origin
            d = mr.dirx
            dir_mv = o + d*10
            v_dir = geomie3d.create.vertex(dir_mv)
            v = geomie3d.create.vertex(o)
            vs.append(v)
            dir_e = geomie3d.create.pline_edge_frm_verts([v, v_dir])
            dir_es.append(dir_e)

    if len(hbs) != 0:
        hbx = []
        for hb in hbs:
            bx = bbox2box(hb)
            bx = geomie3d.get.edges_frm_solid(bx)
            hbx.extend(bx)

    if len(mbs) != 0:
        mbx = []
        for mb in mbs:
            bx = bbox2box(mb)
            bx = geomie3d.get.edges_frm_solid(bx)
            mbx.extend(bx)

def test_calc_rays_faces_intersect():
    rays_xyz = [[[0,0,1], [0,0,1]],
                [[0,0,1], [1,0,1]],
                [[0,0,1], [0,0,-1]]]

    ray_list = []
    vlist = []
    for ray_xyz in rays_xyz:
        ray = geomie3d.create.ray(ray_xyz[0],ray_xyz[1])
        ray_list.append(ray)
        #create verts for viz
        v = geomie3d.create.vertex(ray_xyz[0])
        vlist.append(v)
        
    attrib_ls = [{'temperature':29},
                {'temperature':33},
                {'temperature':33}]

    box = geomie3d.create.box(10, 10, 10)
    face_list = geomie3d.get.faces_frm_solid(box)
    face_list = np.take(face_list, [5, 3, 0])
    flist2 = []
    for cnt,ff in enumerate(face_list):
        ff = geomie3d.modify.reverse_face_normal(ff)
        ff.overwrite_attributes(attrib_ls[cnt])
        n = geomie3d.get.face_normal(ff)
        flist2.append(ff)

    hrays,mrays,hit_faces,miss_faces = geomie3d.calculate.rays_faces_intersection(ray_list,
                                                                                flist2)

    if len(hit_faces) != 0:
        edge_ls = []
        for hit_face in hit_faces:
            hf_att = hit_face.attributes['rays_faces_intersection']
            int_pts = hf_att['intersection']
            rays = hf_att['ray']
            for cnt,intpt in enumerate(int_pts):
                xyzs = [intpt, rays[cnt].origin]
                vs = geomie3d.create.vertex_list(xyzs)
                edge = geomie3d.create.pline_edge_frm_verts(vs)    
                edge_ls.append(edge)

    if len(hrays) != 0:
        hit_vlist = []
        for hray in hrays:
            #create verts for viz
            v = geomie3d.create.vertex(hray.origin)
            hit_vlist.append(v)
        
    if len(mrays) != 0:
        miss_vlist = []
        for mray in mrays:
            #create verts for viz
            v = geomie3d.create.vertex(mray.origin)
            intersect_pt = mray.origin + mray.dirx*2
            v2 = geomie3d.create.vertex(intersect_pt)
            edge = geomie3d.create.pline_edge_frm_verts([v,v2])
            miss_vlist.append(v)
            miss_vlist.append(edge)

def test_calc_trsf():
    #add an extra column to the points
    xyzs = np.array([[0,0,0], [0,0,1]])
    nxyz = len(xyzs)
    xyzw = np.ones((nxyz,4))
    xyzw[:,:-1] = xyzs

    trst_mat = geomie3d.calculate.translate_matrice(0, 0, 0)
    rot_mat = geomie3d.calculate.rotate_matrice((1,0,0), -90)

    xyz3 = np.dot(xyzw, rot_mat.T)

    #compose matrix
    compose = trst_mat@rot_mat
    xyz4 = np.dot(xyzw, compose.T)

    #inverse the matrix to return the point to normal
    rot_mat_inv = np.linalg.inv(rot_mat)
    xyz4 = np.dot(xyz3, rot_mat_inv.T)

    #remove the last column of the points
    trsf_xyzs = xyz4[:,:-1]

    box = geomie3d.create.box(1, 1, 1)
    trsl_mat = geomie3d.calculate.translate_matrice(1, 1, 0)
    rot_mat = geomie3d.calculate.rotate_matrice((0,0,1), -60)
    trsl_mat2 = geomie3d.calculate.inverse_matrice(trsl_mat)

    verts = geomie3d.get.vertices_frm_solid(box)
    xyzs = [v.point.xyz for v in verts]
    # trsf_xyzs = geomie3d.calculate.trsf_xyzs(xyzs, rot_mat@trsl_mat)
    trsf_xyzs = geomie3d.calculate.trsf_xyzs(xyzs, rot_mat)
    cnt = 0
    for v in verts: 
        v.point.xyz = trsf_xyzs[cnt] 
        cnt+=1

def test_calculate_angle_btw_2vectors():
    v1 = [1,0,0]
    v2 = [-1,1,0]
    z = geomie3d.calculate.cross_product(v1,v2)
    angle = geomie3d.calculate.angle_btw_2vectors(v1, v2)

    s = np.array([2.23234, -18.47, 0.0])
    e = np.array([11.86, -17.27,  0.0])
    v3 = e-s
    v3 = geomie3d.calculate.normalise_vectors([v3])[0]
    v3 = np.round(v3, decimals=4)

    angle1 = geomie3d.calculate.angle_btw_2vectors(v1, v3)
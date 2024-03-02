import geomie3d
import numpy as np

def test_create_bboxes_frm_pts():
    midpts = np.array([[0,0,0], [5,5,5]])
    xdims = np.array([5,5])
    ydims = np.array([5,5])
    zdims = np.array([5,5])
    att_ls = [{'id': 0}, {'id': 1}]
    bboxes = geomie3d.create.bboxes_frm_midpts(midpts, xdims, ydims, zdims, attributes_list=att_ls)
    boxes = geomie3d.create.boxes_frm_bboxes(bboxes)

    lwr_left_pts = np.array([[0,0,0], [5,5,5]])
    xdims = np.array([5,5])
    ydims = np.array([5,5])
    zdims = np.array([0,0])
    att_ls = [{'id': 0}, {'id': 1}]
    bboxes = geomie3d.create.bboxes_frm_lwr_left_pts(lwr_left_pts, xdims, ydims, zdims, attributes_list=att_ls)
    boxes1 = geomie3d.create.boxes_frm_bboxes(bboxes)

def test_create_box():
    box = geomie3d.create.box(10, 5, 1)
    es = geomie3d.get.edges_frm_solid(box)
    shell = box.shell
    face_list = shell.face_list

    for face in face_list:
        wire = face.bdry_wire
        vlist = geomie3d.get.points_frm_wire(wire)
        tri_face = geomie3d.modify.triangulate_face(face)
        n = geomie3d.get.face_normal(face)
    
    vlist = geomie3d.get.topo_explorer(box, geomie3d.topobj.TopoType.VERTEX)
    pts = [geomie3d.get.point_frm_vertex(v) for v in vlist]

def test_create_bspline_edge():
    ctrl_pts = [[0,0,0], [0,20,0], [10,0,0], [20,0,0]]
    e = geomie3d.create.bspline_edge_frm_xyzs(ctrl_pts, degree=2, resolution=0.01)
    vs3 = geomie3d.create.vertex_list(ctrl_pts)
    edge1 = geomie3d.create.pline_edge_frm_verts(vs3)
    # Get curve points
    points = e.curve.evaluate_single(0.5)
    points = e.curve.evalpts
    vs = geomie3d.create.vertex_list(points)
    edge = geomie3d.create.pline_edge_frm_verts(vs)
    vs2 = geomie3d.get.vertices_frm_edge(edge)

def create_bspline_face():
    #using two edges and lofting them
    ctrlpts1 = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [1,-1,0], [0, 0, 0]]
    e1 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts1, degree=2, resolution=0.01, 
                                                    attributes = {})

    ctrlpts2 = [[0, 0, 2], [1, 1, 2], [2, 0, 2], [1,-1,2], [0, 0, 2]]
    e2 = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts2, degree=2, resolution=0.01, 
                                            attributes = {})

    elist = [e1,e2]
    f = geomie3d.create.bspline_face_frm_loft(elist)
    g = geomie3d.create.grids_frm_bspline_face(f, 4, 4)
    v = geomie3d.create.grid_pts_frm_bspline_face(f, 15, 15)

    # using control points
    ctrl_pts = [[-25,-25,-5],[-25,-15,0],[-25,-5,0],[-25,5,0],[-25,15,0],[-25,25,5], 
                [-15,-25,0], [-15,-15,0],[-15,-5,0],[-15,5,0],[-15,15,1],[-15,25,1], 
                [-5,-25,5],[-5,-15,5],[-5,-5,5],[-5,5,0],[-5,15,0],[-5,25,0], 
                [5,-25,0],[5,-15,0],[5,-5,0],[5,5,-5],[5,15,-5],[5,25,-5], 
                [15,-25,0],[15,-15,0],[15,-5,0],[15,5,0],[15,15,-10],[15,25,-10],
                [25,-25,0], [25,-15,0], [25,-5,0],[25,5,0],[25,15,0],[25,25,-5]]
    
    deg_u = 1
    deg_v = 1

    kv_u = 6
    kv_v = 6

    f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v, 
                                                resolution=0.167)
    vs1 = geomie3d.create.vertex_list(ctrl_pts)
    surf_pts = f.surface.evalpts
    vs2 = geomie3d.create.vertex_list(surf_pts)

def test_create_composite():
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
    face = geomie3d.create.polygon_face_frm_wires(bdry_wire, [hole_wire], 
                                                attributes = f_att)

    topo_list = []
    topo_list.extend(vlist1)
    topo_list.append(pline_edge1)
    topo_list.append(pline_edge2)
    topo_list.append(bdry_wire)
    topo_list.append(face)
    cmp = geomie3d.create.composite(topo_list)

    topo_list.append(cmp)
    cmp2 = geomie3d.create.composite(topo_list)

    topo_list.append(cmp2)
    cmp3 = geomie3d.create.composite(topo_list)
    d = cmp3.sorted2dict()

    d1 = geomie3d.get.unpack_composite(cmp2)

def test_create_edge():
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

    #create bspline edge
    ctrlpts = [[0, 0, 0], [1, 1, 0], [2, 0, 0], [1,-1,0], [0,0,0]]
    bspline_edge = geomie3d.create.bspline_edge_frm_xyzs(ctrlpts, degree=2, resolution=0.01)
    vs = geomie3d.get.vertices_frm_edge(bspline_edge)

    f = geomie3d.create.polygon_face_frm_verts(vs)

def test_create_extrude():
    xyz_list1 = [[10,10,0], [20,10,0], [20,20,0], [10,20,0]]
    hole_xyzs = [[12,12,0], [12,18,0], [18,18,0], [18,12,0]]
    xyz_list1.reverse()
    vlist1 = geomie3d.create.vertex_list(xyz_list1)
    hole_vertices = geomie3d.create.vertex_list(hole_xyzs)
    f = geomie3d.create.polygon_face_frm_verts(vlist1, hole_vertex_list=[hole_vertices])
    solid = geomie3d.create.extrude_polygon_face(f, [0,-1,-1], 20)

def test_create_face():
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
    face = geomie3d.create.polygon_face_frm_wires(bdry_wire, [hole_wire], 
                                                attributes = f_att)

def test_create_grid3d_bboxes():
    bbox = geomie3d.utility.Bbox([1,1,5,10,10,10])
    div_bboxes = geomie3d.create.grid3d_from_bbox(bbox, 5, 5, 5)

    big_box = geomie3d.create.boxes_frm_bboxes([bbox])
    boxes = geomie3d.create.boxes_frm_bboxes(div_bboxes)

def test_create_nrml_edges_frm_face():
    bx = geomie3d.create.box(10,10,10)
    faces = geomie3d.get.faces_frm_solid(bx)
    edges = geomie3d.create.pline_edges_frm_face_normals(faces, magnitude=5)

def test_create_vertices():
    v = geomie3d.create.vertex((0,0,0), attributes = {"name": "special_point"})
    a = v.attributes
    p = v.point
    coord = p.xyz

    xyz_list = [(0,0,0), (10,0,0), (10,10,0), (0,10,0)]
    att_list = [{"x":1}, {"x":2}, {"x":3}, {"x":4}]
    vlist = geomie3d.create.vertex_list(xyz_list, attributes_list = att_list)

def test_create_wire():
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
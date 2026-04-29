import geomie3d
import geomie3d.viz

def get_cs_frm_face(face):
    o = geomie3d.calculate.face_midxyz(face)
    face_verts = geomie3d.get.vertices_frm_face(face)
    xyzs = [fv.point.xyz for fv in face_verts]
    xd = xyzs[3] - o
    xd = geomie3d.calculate.normalise_vectors(xd)
    yd = xyzs[5] - o
    yd = geomie3d.calculate.normalise_vectors(yd)
    cs = geomie3d.create.coordinate_system_frm_arrs(o, xd, yd)
    return cs

def viz_cs(face):
    n = geomie3d.get.face_normal(face)
    o = geomie3d.calculate.face_midxyz(face)
    o_vert = geomie3d.create.vertex(o)
    face_verts = geomie3d.get.vertices_frm_face(face)
    mv_o = geomie3d.calculate.move_xyzs([o], [n], 5)[0]
    mv_o_vert = geomie3d.create.vertex(mv_o)
    z_edge = geomie3d.create.pline_edge_frm_verts([o_vert, mv_o_vert])
    x_edge = geomie3d.create.pline_edge_frm_verts([o_vert, face_verts[3]])
    y_edge = geomie3d.create.pline_edge_frm_verts([o_vert, face_verts[5]])
    # geomie3d.viz.viz([{'topo_list': [z_edge, y_edge, x_edge], 'colour': 'green'}])
    return x_edge, y_edge, z_edge

# create orig cs
face_xyzs = [[1, 1, 0],
             [3.5, 1, 0],
             [6, 1, 0],
             [6, 3.5, 0],
             [6, 6, 0],
             [3.5, 6, 0],
             [1, 6, 0],
             [1, 3.5, 0]]

face_verts = geomie3d.create.vertex_list(face_xyzs)
face = geomie3d.create.polygon_face_frm_verts(face_verts)
rot_matx = geomie3d.calculate.rotate_matrice([1, 0, 0], 45)
rot_maty = geomie3d.calculate.rotate_matrice([0, 1, 0], 45)
face = geomie3d.modify.trsf_topos([face], [rot_matx@rot_maty])[0]
cs1 = get_cs_frm_face(face)
print(cs1.x_dir, cs1.y_dir, cs1.origin)
xe1, ye1, ze1 = viz_cs(face)

# create dest cs
rot_matx = geomie3d.calculate.rotate_matrice([1, 0, 0], 45)
rot_maty = geomie3d.calculate.rotate_matrice([0, 1, 0], 10)
trsl_mat = geomie3d.calculate.translate_matrice(3, 4, 5)
trsf_mat = trsl_mat@rot_maty@rot_matx
trsf_face = geomie3d.modify.trsf_topos([face], [trsf_mat])[0]
cs2 = get_cs_frm_face(trsf_face)
xe2, ye2, ze2 = viz_cs(trsf_face)
trsf_mat  = geomie3d.calculate.cs2cs_matrice(cs1, cs2)
trsf_mat_inv = geomie3d.calculate.inverse_matrice(trsf_mat)
trsf_topo = geomie3d.modify.trsf_topo_based_on_cs(face, cs1, cs2)
trsf_topo_inv = geomie3d.modify.trsf_topos([trsf_topo], [trsf_mat_inv])[0]

geomie3d.viz.viz([{'topo_list': [trsf_topo], 'colour': 'green'},
                  {'topo_list': [trsf_topo_inv], 'colour': 'orange'},
                  {'topo_list': [face], 'colour': 'blue'},
                  {'topo_list': [trsf_face], 'colour': 'blue'},
                  {'topo_list': [ze1], 'colour': 'red'},
                  {'topo_list': [xe2, ye2, ze2], 'colour': 'red'}])
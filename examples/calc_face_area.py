import geomie3d
import geomie3d.viz

# poly_xyzs = [[0, 0, 0], [10, 0, 0], [10, 3, 4], [0, 3, 4]]
poly_xyzs = [[0,10,0], [0,0,0], [5,0,0], [5,2,0], [3,2,0], [3,10,0]]

vs = geomie3d.create.vertex_list(poly_xyzs)
face = geomie3d.create.polygon_face_frm_verts(vs)
n = geomie3d.get.face_normal(face)
print(n)
area = geomie3d.calculate.face_area(face)
pline_n = geomie3d.create.pline_edges_frm_face_normals([face])
print(area)
geomie3d.viz.viz([{'topo_list': [face], 'colour':'blue'},
                  {'topo_list': pline_n, 'colour':'blue'}])

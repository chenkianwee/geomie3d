import geomie3d
import geomie3d.viz

poly_xyzs = [[1,1,0], [2,1,0], [2,3,0], 
             [3,3,0], [3,1,0], [4,1,0],
             [4,5,0], [1,5,0]]
vs = geomie3d.create.vertex_list(poly_xyzs)
poly = geomie3d.create.polygon_face_frm_verts(vs)
rev_f = geomie3d.modify.reverse_face_normal(poly)
n = geomie3d.get.face_normal(poly)
n_rev = geomie3d.get.face_normal(rev_f)
print(n, n_rev)
is_ccw = geomie3d.calculate.is_anticlockwise(poly_xyzs, n_rev)
print(is_ccw)
geomie3d.viz.viz([{'topo_list': [poly], 'colour': 'blue'}])
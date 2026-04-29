import geomie3d
import geomie3d.viz
import numpy 

box = geomie3d.create.box(1, 1, 1)
trsl_mat = geomie3d.calculate.translate_matrice(1, 1, 0)
rot_mat = geomie3d.calculate.rotate_matrice((0,0,1), 60.0)
trsl_mat2 = geomie3d.calculate.inverse_matrice(trsl_mat)

verts = geomie3d.get.vertices_frm_solid(box)
xyzs = [v.point.xyz for v in verts]
# print(xyzs)
# trsf_xyzs = geomie3d.calculate.trsf_xyzs(xyzs, trsl_mat@rot_mat)
trsf_xyzs = geomie3d.calculate.trsf_xyzs(xyzs, rot_mat@trsl_mat)
# trsf_xyzs = geomie3d.calculate.trsf_xyzs(xyzs, rot_mat)
# print(trsf_xyzs)
cnt = 0
for v in verts: 
    v.point.xyz = trsf_xyzs[cnt] 
    cnt+=1

bx_face = geomie3d.get.faces_frm_solid(box)
for bf in bx_face:
    bf.update_polygon_surface()

box2 = geomie3d.create.box(1, 1, 1)
geomie3d.viz.viz([{'topo_list':[box], 'colour': 'white'}, {'topo_list':[box2], 'colour': 'red'}])

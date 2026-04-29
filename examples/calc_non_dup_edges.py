import geomie3d
import geomie3d.viz

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
print(dup2)
geomie3d.viz.viz([{'topo_list':outline_ls, 'colour': 'blue'}])

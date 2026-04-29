import geomie3d
import geomie3d.viz

box = geomie3d.create.box(10, 5, 1)
es = geomie3d.get.edges_frm_solid(box)
shell = box.shell
face_list = shell.face_list

for face in face_list:
    wire = face.bdry_wire
    vlist = geomie3d.get.points_frm_wire(wire)
    tri_face = geomie3d.modify.triangulate_face(face)
    n = geomie3d.get.face_normal(face)
    print(n)
    geomie3d.viz.viz([{'topo_list': [face], 'colour':'blue'},
                      {'topo_list': es, 'colour':'white'}])
    # print(tri_face)
    # print([v.xyz for v in vlist])
   
vlist = geomie3d.get.topo_explorer(box, geomie3d.topobj.TopoType.VERTEX)
pts = [geomie3d.get.point_frm_vertex(v) for v in vlist]
geomie3d.viz.viz([{'topo_list':[box], 'colour': 'red'}])

# print(pts)
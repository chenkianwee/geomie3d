import geomie3d
import geomie3d.viz

xyz_list1 = [[10,10,0], [20,10,0], [20,20,0], [10,20,0]]
hole_xyzs = [[12,12,0], [12,18,0], [18,18,0], [18,12,0]]
xyz_list1.reverse()
vlist1 = geomie3d.create.vertex_list(xyz_list1)
hole_vertices = geomie3d.create.vertex_list(hole_xyzs)
f = geomie3d.create.polygon_face_frm_verts(vlist1, hole_vertex_list=[hole_vertices])
solid = geomie3d.create.extrude_polygon_face(f, [0,-1,-1], 20)
geomie3d.viz.viz([{'topo_list':[solid], 'colour':'blue'}])
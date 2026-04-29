import geomie3d
import geomie3d.viz

polys = [[[1,1,0], [5,1,0], [5,5,0], [1,5,0]], 
         [[5,5,0], [7,6,0], [8,5,0], [8,8,0], [5,8,0]],
         [[20,0,0], [20,5,0], [10,5,0], [10,10,0], [0,10,0], [0,0,0]]]

faces = []
for poly in polys:
    vlist = geomie3d.create.vertex_list(poly)
    f = geomie3d.create.polygon_face_frm_verts(vlist)
    faces.append(f)

are_convex = geomie3d.calculate.are_polygon_faces_convex(faces)
print(are_convex)
geomie3d.viz.viz([{'topo_list': faces, 'colour': 'blue'}])
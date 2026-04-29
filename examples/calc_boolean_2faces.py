import geomie3d
import geomie3d.viz
import numpy as np

def xyzs2faces(poly):
    polyvs = geomie3d.create.vertex_list(poly)
    f = geomie3d.create.polygon_face_frm_verts(polyvs)
    return f

# defining the boundary wire 1
# poly1 = [[10.0, 10.0, 0], [20, 10, 0], [20, 20, 0], [10, 20, 0]]
poly1 = [[10.0, 10.0, 5], [20, 10, 5], [20, 20, 5], [10, 20, 5]]

# poly2 = [[15, 5, 0], [25, 5, 0], [25, 15, 0], [15, 15, 0]]
poly2 = [[15, 5, 5], [20, 3, 5], [25, 5, 5], [25, 15, 5], [15, 15, 5], [11, 12, 5]]
# poly2 = [[11, 5, 5], [19, 5, 5], [19, 12, 5], [16, 12, 5], [16, 8, 5], [14, 8, 5], [14, 12, 5], [11, 12, 5]]
poly2 = [[11, 5, 5], [19, 5, 5], [19, 12, 5], [15, 12, 5], [15, 11, 5], [16, 11, 5], [16, 11.5, 5], [18, 11.5, 5],
         [18, 8, 5], [14, 8, 5], [14, 12, 5], [11, 12, 5]]

face1 = xyzs2faces(poly1)
face2 = xyzs2faces(poly2)
res_faces = geomie3d.calculate.polygons_clipping(face1, face2, 'subject_not_clip')
print(res_faces)
xyz_ls = []
for f in res_faces:
    vs = geomie3d.get.vertices_frm_face(f)
    xyzs = np.array([v.point.xyz for v in vs])
    xyz_ls.append(xyzs)

xyz_ls = np.array(xyzs)

correct_answer = np.array([[14, 10,  5,], [14,  8,  5,], [18,  8,  5], [18, 10,  5], 
                           [19, 10,  5], [19,  5,  5], [11,  5,  5], [11, 10,  5]]).astype(float)
print(xyz_ls)
assert np.array_equiv(xyz_ls, correct_answer) 

edges1 = geomie3d.get.edges_frm_face(face1)
edges2 = geomie3d.get.edges_frm_face(face2)
geomie3d.viz.viz([{'topo_list': edges1, 'colour': 'red'},
                  {'topo_list': edges2, 'colour': 'green'}])

if res_faces != None:
    geomie3d.viz.viz([{'topo_list': res_faces, 'colour': 'blue'},
                      {'topo_list': edges1, 'colour': 'red'},
                      {'topo_list': edges2, 'colour': 'green'}])

    geomie3d.viz.viz([{'topo_list': res_faces, 'colour': 'blue'}])
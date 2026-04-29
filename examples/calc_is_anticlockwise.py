import geomie3d
import numpy as np

# xyzs = [[2,6,0], [2,1,0], [3,3,0]]
# xyzs = [[2,6,0], [2,1,0], [2,5,0], [2,9,0]]
# xyzs = [[[2,6,0], [2,1,0], [3,3,0]],
#         [[2,6,0], [2,1,0], [1,3,0]],
#         [[2,6,0], [2,1,0], [2,3,0]]]

xyzs = [[11, 5, 5], [19, 5, 5], [19, 12, 5], [15, 12, 5], [15, 11, 5], [16, 11, 5], [16, 11.5, 5], [18, 11.5, 5],
        [18, 8, 5], [14, 8, 5], [14, 12, 5], [11, 12, 5]]

verts = geomie3d.create.vertex_list(xyzs)
face = geomie3d.create.polygon_face_frm_verts(verts)
n = geomie3d.get.face_normal(face)
print(n)
# xyzs = np.flip(xyzs, axis=0)
ccw = geomie3d.calculate.is_anticlockwise(xyzs, n)
print(ccw)

# xyzs = [[1,1,0], [1,5,0], [2,3,0]]
xyzs = [[0,1,0], [1,1,0], [1,5,0], [0,5,0], [1.5,1.5,0]]
# xyzs = [[0,1,0], [1,1,0], [1,5,0], [0,5,0]]
wnum = geomie3d.calculate.winding_number([xyzs], [0,0,1])
print(wnum)


import geomie3d
import numpy as np
xyzs1 = [[[0,0,0], [3,0,0],[2,3,0],[0,3,1], [1,3,0], [4,3,0]],
        [[0,0,0], [3,0,0],[2,3,0],[1,3,0], [2,4,0], [0,4,0]]]

xyzs2 = [[[0,0,0], [3,0,0],[2,3,0],[0,3,1], [1,3,0], [4,3,0]],
        [[0,0,0], [3,0,0],[2,3,0],[1,3,0], [2,4,0]]]

xyzs3 = [[0,0,0], [3,0,0], [2,3,0], [0,3,0], [0,3,0], [0,0,0]]

xyzs4 = [[0,0,0], [0,10,0], [0,20,1]]

vs1 = [geomie3d.create.vertex_list(xyz) for xyz in xyzs1]
vs2 = [geomie3d.create.vertex_list(xyz) for xyz in xyzs2]
vs3 = [geomie3d.create.vertex(xyz) for xyz in xyzs3]
vs4 = [geomie3d.create.vertex(xyz) for xyz in xyzs4]

is_coplanar = geomie3d.calculate.is_coplanar_xyzs(xyzs1)
print(is_coplanar)

is_coplanar = geomie3d.calculate.is_coplanar_xyzs(xyzs2)
print(is_coplanar)

is_coplanar = geomie3d.calculate.is_coplanar_xyzs(xyzs3)
print(is_coplanar)

is_coplanar = geomie3d.calculate.is_coplanar(vs1)
print(is_coplanar)
is_coplanar = geomie3d.calculate.is_coplanar(vs2)
print(is_coplanar)
is_coplanar = geomie3d.calculate.is_coplanar(vs3)
print(is_coplanar)

centre_pt = geomie3d.calculate.xyzs_mean(xyzs1)
cps = np.array([[1.66666667, 2.0, 0.16666667],
                [1.33333333, 2.33333333, 0.]])

print(centre_pt)

is_collinear = geomie3d.calculate.is_collinear(vs4)
print(is_collinear)
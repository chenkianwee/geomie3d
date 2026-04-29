import numpy as np
import geomie3d

xyz_2dlist = [[[0,0,0], [0,0,1], [1,0,1]],
              [[0,0,0], [0,0,1]],
              [[0,0,0], [0,0,1]]]

trst_mat = geomie3d.calculate.translate_matrice(10, 0, 0)
trst_mat2 = geomie3d.calculate.translate_matrice(0, 10, 0)
rot_mat = geomie3d.calculate.rotate_matrice((1,0,0), -90)

trsf_xyzs = geomie3d.calculate.trsf_xyzs(xyz_2dlist, [trst_mat, trst_mat2, rot_mat])
print(trsf_xyzs)
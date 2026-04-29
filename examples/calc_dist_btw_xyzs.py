import numpy as np
import geomie3d

xyz1 = [[0,0,0], [1,1,2], [3,3,3]]
xyz2 = [0,0,0]
dist = geomie3d.calculate.dist_btw_xyzs(xyz1, xyz2)
# print(dist)

xyz1 = np.array([2.23912, -19.36, 0])
xyz2 = np.array([11.86, -18.15, 0])
vec1 = xyz2 - xyz1
norm = geomie3d.calculate.normalise_vectors([vec1])
print(norm)
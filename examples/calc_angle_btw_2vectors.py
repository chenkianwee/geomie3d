import geomie3d
import numpy as np

v1 = [1,0,0]
v2 = [-1,1,0]
z = geomie3d.calculate.cross_product(v1,v2)
angle = geomie3d.calculate.angle_btw_2vectors(v1, v2)
# print(z)
# print(angle)

s = np.array([2.23234, -18.47, 0.0])
e = np.array([11.86, -17.27,  0.0])
v3 = e-s
v3 = geomie3d.calculate.normalise_vectors([v3])[0]
v3 = np.round(v3, decimals=4)
# print(v3)

angle1 = geomie3d.calculate.angle_btw_2vectors(v1, v3)
# print(angle1)

v1 = [0,0,1]
v2 = [1,0,-0.5]
angle = geomie3d.calculate.angle_btw_2vectors(v1, v2)
print(angle)
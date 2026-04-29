import geomie3d

import numpy as np

v = geomie3d.create.vertex((0,0,0), attributes = {"name": "special_point"})
a = v.attributes
p = v.point
coord = p.xyz

xyz_list = [(0,0,0), (10,0,0), (10,10,0), (0,10,0)]
att_list = [{"x":1}, {"x":2}, {"x":3}, {"x":4}]
vlist = geomie3d.create.vertex_list(xyz_list, attributes_list = att_list)
print(np.array([vert.attributes for vert in vlist]))
print(vlist)
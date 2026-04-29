import geomie3d

linexyzs = [[[1,0,0], [1,6,0]], 
            [[2,2,0], [2,8,0]]]
ts = [1.5, 0.5]

xyzs = geomie3d.calculate.linexyzs_from_t(ts, linexyzs)
print(xyzs)

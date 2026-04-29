import numpy as np
import geomie3d
import geomie3d.viz

xyzs = [[8,3,-2.5], [8,4,2]]
# xyzs = [[2,3,-2.5], [4,2,0]]
polyxyzs = [[[1,1,0], [5,1,0], [5,5,0], [1,5,0]], [[1,1,1], [5,1,1], [5,5,1]]]

nrmlxyzs = [[0,0,1], [0,0,1]]
dist2polys, polyintxs = geomie3d.calculate.dist_pointxyzs2polyxyzs(xyzs, polyxyzs, nrmlxyzs, int_pts=True)
print('results', dist2polys)
print('results', polyintxs)

# polyxyzs = [[[1,1,0], [5,1,0], [5,5,0]], [[1,1,0], [5,1,0], [8,1,0]]]
# pls = geomie3d.calculate.planes_frm_3pts(polyxyzs)
# print(pls)
# pl = geomie3d.utility.Plane([1,0,1], [0,1,1])
# print(pl.a, pl.b, pl.c, pl.d)

verts = geomie3d.create.vertex_list(xyzs)
polys = []
for polyxyz in polyxyzs:
    poly_verts = geomie3d.create.vertex_list(polyxyz)
    poly = geomie3d.create.polygon_face_frm_verts(poly_verts)
    polys.append(poly)

dists, intxs = geomie3d.calculate.dist_verts2polyfaces(verts, polys, int_pts=True)
intxyzs = np.array([v.point.xyz for v in intxs])
print('results',dists, intxyzs)

# assert np.allclose(dist2polys, np.array([3.90512484, 3.16227766]))
# assert np.array_equal(intxyzs, np.array([[5,3,0], [5,4,1]]))

# intverts = geomie3d.create.vertex_list(polyintxs)
edges = []
for i in range(len(verts)):  
    edge = geomie3d.create.pline_edge_frm_verts([verts[i], intxs[i]])
    edges.append(edge)

geomie3d.viz.viz([{'topo_list': polys, 'colour': 'blue'},
                  {'topo_list': verts, 'colour': 'red'},
                  {'topo_list': intxs, 'colour': 'green'},
                  {'topo_list': edges, 'colour': 'red'}
                  ])
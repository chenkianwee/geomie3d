import geomie3d
import geomie3d.viz

from dateutil.parser import parse

# the geometry data to view
ctrl_pts = [[1,5,0], [5,5,0],
            [1,0,0], [5,0,0]]

deg_u = 1
deg_v = 1

kv_u = 2
kv_v = 2

f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v)
g = geomie3d.create.grids_frm_bspline_face(f, 5, 5)

bx = geomie3d.create.box(5, 5, 5)
edges1 = geomie3d.get.edges_frm_solid(bx)
bx2 = geomie3d.create.box(10, 10, 5)
edges2 = geomie3d.get.edges_frm_solid(bx2)
bx3 = geomie3d.create.box(15, 15, 10)
edges3 = geomie3d.get.edges_frm_solid(bx3)

res = []
for i in range(len(g)):
    res.append(i*10)

res2 = []
for i in range(len(g)):
    res2.append(i*2)

res3 = []
for i in range(len(g)):
    res3.append(i*5)

topo2d = [g, g, g]
res2d = [res, res2, res3]
topo_datetime_ls = [parse('2023-02-15T13:51'), parse('2023-02-15T12:51'), parse('2023-02-15T14:51')]
topo_2ddlist = [[{'topo_list': edges1, 'colour': 'red'}], 
                [{'topo_list': edges2, 'colour': 'green'}], 
                [{'topo_list': edges3, 'colour': 'blue'}]]

geomie3d.viz.viz_animate_falsecolour(topo2d, res2d, topo_datetime_ls, false_min_max_val=[0,240], other_topo_2ddlist=topo_2ddlist)
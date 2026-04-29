import geomie3d
import geomie3d.viz

pointxyzs = [[1,6,1], [8,2,0]]
linexyzs = [[[1,6,0], [5,6,0]],
            [[1,6,0], [5,6,2]]]

vlist = geomie3d.create.vertex_list(pointxyzs)

edge_ls = []
for linexyz in linexyzs:
    verts = geomie3d.create.vertex_list(linexyz)
    edge = geomie3d.create.pline_edge_frm_verts(verts)
    edge_ls.append(edge)
    
dists, int_vs = geomie3d.calculate.dist_vertex2line_edge(vlist, edge_ls, int_pts = True)

print(dists)
print(int_vs)
# draw the edge between the point and the closest int_vs for viz
int_es = []
for cnt,v in enumerate(vlist):
    e = geomie3d.create.pline_edge_frm_verts([v, int_vs[cnt]])
    int_es.append(e)

geomie3d.viz.viz([{'topo_list':edge_ls, 'colour':'blue'},
                  {'topo_list':vlist, 'colour':'green'},
                  {'topo_list':int_vs, 'colour':'red'},
                  {'topo_list':int_es, 'colour':'red'}])
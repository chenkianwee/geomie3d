import geomie3d
import geomie3d.viz

linexyzs1 = [[[1,2,0],[6,5,0]],
             [[1,1,0],[1,6,0]],
             [[3,2,0],[6,5,0]]]

linexyzs2 = [[[6,2,10],[1,5,0]],
             [[6,2,0],[2,5,0]],
             [[6,1,0],[6,6,0]],
             ]

# linexyzs1 = [[[20,20,0],[10,20,0]]]

# linexyzs2 = [[[15,15,0],[15,5,0]]]


int_pts = geomie3d.calculate.linexyzs_intersect(linexyzs1, linexyzs2)
print(int_pts)

def line2edge(linexyzs):
    edge_ls = []
    for linexyz in linexyzs:
        vlist = geomie3d.create.vertex_list(linexyz)
        edge = geomie3d.create.pline_edge_frm_verts(vlist)
        edge_ls.append(edge)
    return edge_ls

edge_list1 = line2edge(linexyzs1)
edge_list2 = line2edge(linexyzs2)

int_vs = geomie3d.calculate.lineedge_intersect(edge_list1, edge_list2)
print(int_vs)

geomie3d.viz.viz([{'topo_list': edge_list1, 'colour': 'red'},
                  {'topo_list': edge_list2, 'colour': 'blue'},
                  {'topo_list': int_vs[2:], 'colour': 'green'}])
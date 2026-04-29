import geomie3d
import geomie3d.viz

xyz_list = [[[2,3,0], [5,3,0]],
            [[6,9,0], [4,9,1]],
            [[5,8,0], [2,8,0]],
            [[5,3,0], [5,8,0]],
            [[2,8,0], [2,3,0]],
            [[5,3,0], [6,1,0]],
            [[5,8,0], [6,9,0]],
            ]

# xyz_list = [[[5,3,0], [6,1,0]],
#             [[2,3,0], [5,3,0]],
#             [[6,9,0], [4,9,1]],
#             [[5,8,0], [2,8,0]],
#             [[5,3,0], [5,8,0]],
#             [[5,8,0], [6,9,0]],
#             ]

edge_list = []
for cnt,xyzs in enumerate(xyz_list):
    vertices = geomie3d.create.vertex_list(xyzs)
    e = geomie3d.create.pline_edge_frm_verts(vertices, attributes = {'id': cnt})
    edge_list.append(e)

connected_indxs = geomie3d.calculate.a_connected_path_from_edges(edge_list, indx=True)
print(connected_indxs)
connected = geomie3d.calculate.a_connected_path_from_edges(edge_list, indx=False)

geomie3d.viz.viz([{'topo_list': connected['connected'], 'colour': 'blue', 'attribute': 'id'},
                  {'topo_list': connected['loose'], 'colour': 'red', 'attribute': 'id'}])

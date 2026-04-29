import geomie3d

# xyz_2dlist = [[[3,3,0], [3,0.5,0]]]
# polyxyzs = [[[1,1,0], [5,1,0], [5,5,0], [1,5,0]]]
# are_xyzs = geomie3d.calculate.are_xyzs_in_polyxyzs(xyz_2dlist, polyxyzs)

# vs = geomie3d.create.vertex_list([[3,3,0], [3,0.5,0]])
# polyvs = geomie3d.create.vertex_list([[1,1,0], [5,1,0], [5,5,0], [1,5,0]])
# poly = geomie3d.create.polygon_face_frm_verts(polyvs)

# are_xyzs2 = geomie3d.calculate.are_verts_in_polygons([vs, vs], [poly, poly])

# assert are_xyzs2 == [[True, False], [True, False]]

# geomie3d.viz.viz([{'topo_list': [poly], 'colour': 'blue'},
#                   {'topo_list': vs, 'colour': 'red'}])

xyz = [-0.03, 6.14333333, 2.63]
polyxyzs = [[-0.03, 1.39, 0.9], [-0.03, 3.22, 0.9], [-0.03, 3.22, 2.42], [-0.03, 1.39, 2.42]]
are_in = geomie3d.calculate.are_xyzs_in_polyxyzs([[xyz]], [polyxyzs])

print(are_in)



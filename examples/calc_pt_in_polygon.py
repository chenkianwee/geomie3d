import geomie3d
import geomie3d.viz

xyz_2dlist = [[[3,2,0], [5,5,0]],
              [[3,4,0], [7,5,0], [7,7,1]],
              [[3,8,0], [4,6,0]]]

polys = [[[1,1,0], [5,1,0], [5,5,0], [1,5,0]], 
         [[5,5,0], [7,3,0], [8,5,0], [8,8,0], [5,8,0]],
         [[1,6,0], [3,7,0], [5,6,0], [5,10,0], [1,10,0]]]


in_polys = geomie3d.calculate.are_xyzs_in_polyxyzs(xyz_2dlist, polys)
print(in_polys)

# for viz
vs1 = geomie3d.create.vertex_list(xyz_2dlist[0])
vs2 = geomie3d.create.vertex_list(xyz_2dlist[1])
vs3 = geomie3d.create.vertex_list(xyz_2dlist[2])
elist = []
for polyxyz in polys:
    vls = geomie3d.create.vertex_list(polyxyz)
    f = geomie3d.create.polygon_face_frm_verts(vls)
    edges = geomie3d.get.edges_frm_face(f)
    elist.append(edges)

geomie3d.viz.viz([{'topo_list': elist[0], 'colour': 'red'},
                  {'topo_list': elist[1], 'colour': 'blue'},
                  {'topo_list': elist[2], 'colour': 'green'},
                  {'topo_list': vs1, 'colour': 'red'},
                  {'topo_list': vs2, 'colour': 'blue'},
                  {'topo_list': vs3, 'colour': 'green'}])
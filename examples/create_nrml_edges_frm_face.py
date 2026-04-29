import geomie3d
import geomie3d.viz

bx = geomie3d.create.box(10,10,10)
faces = geomie3d.get.faces_frm_solid(bx)
edges = geomie3d.create.pline_edges_frm_face_normals(faces, magnitude=5)
geomie3d.viz.viz([{'topo_list':[bx], 'colour': 'blue'},
                  {'topo_list':edges, 'colour': 'red'}])
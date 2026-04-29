import geomie3d
import geomie3d.viz

import numpy as np
rays_xyz = [[[0,0,1], [0,0,1]],
            [[0,0,1], [1,0,1]],
            [[0,0,1], [0,0,-1]]]

ray_list = []
vlist = []
for ray_xyz in rays_xyz:
    ray = geomie3d.create.ray(ray_xyz[0],ray_xyz[1])
    ray_list.append(ray)
    #create verts for viz
    v = geomie3d.create.vertex(ray_xyz[0])
    vlist.append(v)
    
attrib_ls = [{'temperature':29},
             {'temperature':33},
             {'temperature':33}]

box = geomie3d.create.box(10, 10, 10)
face_list = geomie3d.get.faces_frm_solid(box)
face_list = np.take(face_list, [5, 3, 0])
flist2 = []
for cnt,ff in enumerate(face_list):
    ff = geomie3d.modify.reverse_face_normal(ff)
    ff.overwrite_attributes(attrib_ls[cnt])
    n = geomie3d.get.face_normal(ff)
    flist2.append(ff)

hrays,mrays,hit_faces,miss_faces = geomie3d.calculate.rays_faces_intersection(ray_list,
                                                                              flist2)

print(hit_faces)
print(miss_faces)                                                               
print(hrays)
print(mrays)

vis_ls = []
if len(hit_faces) != 0:
    vis_ls.append({'topo_list': hit_faces, 'colour': 'red'})
    edge_ls = []
    for hit_face in hit_faces:
        hf_att = hit_face.attributes['rays_faces_intersection']
        int_pts = hf_att['intersection']
        rays = hf_att['ray']
        for cnt,intpt in enumerate(int_pts):
            xyzs = [intpt, rays[cnt].origin]
            vs = geomie3d.create.vertex_list(xyzs)
            edge = geomie3d.create.pline_edge_frm_verts(vs)    
            edge_ls.append(edge)
    vis_ls.append({'topo_list': edge_ls, 'colour': 'red'})

if len(miss_faces) != 0:
    vis_ls.append({'topo_list': miss_faces, 'colour': 'white'})

if len(hrays) != 0:
    hit_vlist = []
    for hray in hrays:
        #create verts for viz
        v = geomie3d.create.vertex(hray.origin)
        hit_vlist.append(v)
    vis_ls.append({'topo_list': hit_vlist, 'colour': 'red'})
    
if len(mrays) != 0:
    miss_vlist = []
    for mray in mrays:
        #create verts for viz
        v = geomie3d.create.vertex(mray.origin)
        intersect_pt = mray.origin + mray.dirx*2
        v2 = geomie3d.create.vertex(intersect_pt)
        edge = geomie3d.create.pline_edge_frm_verts([v,v2])
        miss_vlist.append(v)
        miss_vlist.append(edge)
    vis_ls.append({'topo_list': miss_vlist, 'colour': 'white'})

geomie3d.viz.viz(vis_ls)

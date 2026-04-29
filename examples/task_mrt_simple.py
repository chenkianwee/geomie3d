import geomie3d
import geomie3d.viz

import numpy as np

def convert_polygon2bspline_face(poly_face):
    verts = geomie3d.get.vertices_frm_face(poly_face)
    pts = np.array([v.point.xyz for v in verts])
    pts = np.array([pts[2], pts[1], pts[3], pts[0]])
    bface = geomie3d.create.bspline_face_frm_ctrlpts(pts, 2, 2, 1, 1)
    return bface

#setup the scene for raytracing
rm = geomie3d.create.box(4, 4, 3.5)
rm_faces = geomie3d.get.faces_frm_solid(rm)
#assign temperatures to the surfaces
srf_temps = [30, 30, 30, 30, 30, 18]
rm_faces2 = []
for cnt, rm_face in enumerate(rm_faces):
    rm_face = geomie3d.modify.reverse_face_normal(rm_face)
    bface = convert_polygon2bspline_face(rm_face)
    grids = geomie3d.create.grids_frm_bspline_face(bface, 2, 2)
    att = {'temperature': srf_temps[cnt]}
    for grid in grids:     
        geomie3d.modify.update_topo_att(grid, att)
    rm_faces2.extend(grids)

#create the analysis grid 
#take the floor face move it to 0.8m height
flr_face = rm_faces[0]
rm_edges = geomie3d.get.edges_frm_solid(rm)
flr_face = geomie3d.modify.reverse_face_normal(flr_face)
analyse_face = geomie3d.modify.move_topo(flr_face, [0,0,-0.95])
#convert the face to bspline face
analyse_bface = convert_polygon2bspline_face(analyse_face)
aly_grids = geomie3d.create.grids_frm_bspline_face(analyse_bface, 4, 4)
#generate the directions for the ray at each point
ndir = 3600
unitball = geomie3d.d4pispace.tgDirs(ndir)
#create the rays for each analyse pts
aly_vs = []
rays = []
for aly_grid in aly_grids:    
    midpt = geomie3d.calculate.face_midxyz(aly_grid)
    midv = geomie3d.create.vertex(midpt)
    aly_vs.append(midv)
    for dix in unitball.getDirList():
        ray = geomie3d.create.ray(midpt, [dix.x, dix.y, dix.z])        
        rays.append(ray)

rays = np.array(rays)
print('********** Ray Casting **************')
hit_rs, miss_rs , hit_fs, miss_fs = geomie3d.calculate.rays_faces_intersection(rays, rm_faces2)
print('********** Calc MRT **************')
#count the MRT of each point
rays_reshape = np.reshape(rays, (len(aly_grids), ndir))
mrts = []
for spt in rays_reshape:
    ttl_mrt = 0
    for r in spt:
        if 'rays_faces_intersection' in r.attributes:
            att = r.attributes['rays_faces_intersection']
            temp_ls = att['hit_face']
            for temps in temp_ls:
                temp = temps.attributes['temperature']
                ttl_mrt+=temp
            
    mrt = ttl_mrt/len(spt)
    mrts.append(mrt)

print('The MRT at each point are:', mrts)
#=================================================================================================
#for viz
#=================================================================================================
viz_ls = []
if len(hit_fs) > 0:
    #viz the scene
    hit_es = []
    for hit_f in hit_fs:
        hit_e = geomie3d.get.bdry_edges_frm_face(hit_f)
        hit_es.extend(hit_e)
        atts = hit_f.attributes['rays_faces_intersection']
        intx_ls = atts['intersection']
        rays = atts['ray']
        #draw the intersections
        for cnt, intx in enumerate(intx_ls):
            ray = rays[cnt]
            origin = ray.origin
            intx_v = geomie3d.create.vertex_list([ray.origin, intx])
            intx_e = geomie3d.create.pline_edge_frm_verts(intx_v)
            hit_es.append(intx_e)
    
    viz_ls.append({'topo_list': hit_es, 'colour': 'red'})
    
if len(miss_fs) > 0:
    miss_es = []
    for m_f in miss_fs:
        miss_e = geomie3d.get.bdry_edges_frm_face(m_f)
        miss_es.extend(miss_e)
    
    viz_ls.append({'topo_list': miss_es, 'colour': 'white'})
    
if len(hit_rs) > 0:
    hit_r_es = []
    for h_r in hit_rs:
        o = h_r.origin
        d = geomie3d.calculate.move_xyzs([o], [h_r.dirx], [0.3])[0]
        hit_vs = geomie3d.create.vertex_list([o,d])
        hit_r_e = geomie3d.create.pline_edge_frm_verts(hit_vs)
        hit_r_es.append(hit_r_e)
        
    viz_ls.append({'topo_list': hit_r_es, 'colour': 'white'})

if len(miss_rs) > 0:
    miss_r_es = []
    for m_r in miss_rs:
        o = m_r.origin
        d = geomie3d.calculate.move_xyzs([o], [m_r.dirx], [0.3])[0]
        miss_vs = geomie3d.create.vertex_list([o,d])
        miss_r_e = geomie3d.create.pline_edge_frm_verts(miss_vs)
        miss_r_es.append(miss_r_e)
        
    viz_ls.append({'topo_list': miss_r_es, 'colour': 'white'})

if sum(mrts) != 0:
    geomie3d.viz.viz_falsecolour(aly_grids, mrts, other_topo_dlist = [{'topo_list': rm_edges, 'colour': 'blue'}])
else:
    geomie3d.viz.viz(viz_ls)
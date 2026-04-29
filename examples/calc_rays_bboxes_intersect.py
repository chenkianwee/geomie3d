import geomie3d
import geomie3d.viz

rays_xyz = [[[1, 0, 0], [0,0,1]],
            [[4, 3, 10], [1,0,0]]]
bbox_arr_ls = [[5, -1, 5, 8, 3, 10],
                [-2, -2, 4, 2, 2, 12]]

#=============================================================================
def bbox2box(bbox):
    dimx = bbox.maxx - bbox.minx
    dimy = bbox.maxy - bbox.miny
    dimz = bbox.maxz - bbox.minz
    centre_pt = [dimx/2+bbox.minx, dimy/2+bbox.miny, dimz/2+bbox.minz]
    bx = geomie3d.create.box(dimx, dimy, dimz, centre_pt = centre_pt)
    return bx
#=============================================================================

ray_ls = [geomie3d.create.ray(ray_xyz[0], ray_xyz[1]) for ray_xyz in rays_xyz]
bbox_list = [geomie3d.create.bbox_frm_arr(bbox_arr) for bbox_arr in bbox_arr_ls]

inter_res = geomie3d.calculate.rays_bboxes_intersect(ray_ls, bbox_list)
hrs = inter_res[0]
mrs = inter_res[1]
hbs = inter_res[2]
mbs = inter_res[3]

#for viz
viz_dlist = []
dir_es = []
if len(hrs) != 0:
    invs = []
    intes = []
    for hr in hrs:
        o = hr.origin
        d = hr.dirx
        dir_mv = o + d*10
        v_dir = geomie3d.create.vertex(dir_mv)
        v = geomie3d.create.vertex(o)
        invs.append(v)
        dir_e = geomie3d.create.pline_edge_frm_verts([v, v_dir])
        dir_es.append(dir_e)
        
        att = hr.attributes['rays_bboxes_intersection']
        intersects = att['intersection']
        for inter in intersects:
            inv = geomie3d.create.vertex(inter)
            invs.append(inv)
            inte = geomie3d.create.pline_edge_frm_verts([v, inv])
            intes.append(inte)
            
    viz_dlist.append({'topo_list': invs, 'colour': 'red'})
    viz_dlist.append({'topo_list': intes, 'colour': 'red'})
    
if len(mrs) != 0:
    vs = []
    for mr in mrs:
        o = mr.origin
        d = mr.dirx
        dir_mv = o + d*10
        v_dir = geomie3d.create.vertex(dir_mv)
        v = geomie3d.create.vertex(o)
        vs.append(v)
        dir_e = geomie3d.create.pline_edge_frm_verts([v, v_dir])
        dir_es.append(dir_e)
        
    viz_dlist.append({'topo_list': vs, 'colour': 'green'})

if len(hbs) != 0:
    hbx = []
    for hb in hbs:
        bx = bbox2box(hb)
        bx = geomie3d.get.edges_frm_solid(bx)
        hbx.extend(bx)
    
    viz_dlist.append({'topo_list': hbx, 'colour': 'red'})

if len(mbs) != 0:
    mbx = []
    for mb in mbs:
        bx = bbox2box(mb)
        bx = geomie3d.get.edges_frm_solid(bx)
        mbx.extend(bx)
    
    viz_dlist.append({'topo_list': mbx, 'colour': 'green'})

viz_dlist.append({'topo_list': dir_es, 'colour': 'green'})
geomie3d.viz.viz(viz_dlist)
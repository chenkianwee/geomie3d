import geomie3d
import geomie3d.viz
import numpy as np

v_size = 10
ray_orig = [0,0,0]
#-----------------------------------------------------------------------------------------
def convert_polygon2bspline_face(poly_face):
    verts = geomie3d.get.vertices_frm_face(poly_face)
    pts = np.array([v.point.xyz for v in verts])
    pts = np.array([pts[2], pts[1], pts[3], pts[0]])
    bface = geomie3d.create.bspline_face_frm_ctrlpts(pts, 2, 2, 1, 1)
    return bface

def bbox2box(bbox):
    dimx = bbox.maxx - bbox.minx
    dimy = bbox.maxy - bbox.miny
    dimz = bbox.maxz - bbox.minz
    centre_pt = [dimx/2+bbox.minx, dimy/2+bbox.miny, dimz/2+bbox.minz]
    bx = geomie3d.create.box(dimx, dimy, dimz, centre_pt = centre_pt)
    return bx


viz_dlist = []
bx = geomie3d.create.box(20,20,20)
bfaces = geomie3d.get.faces_frm_solid(bx)
bbx_ls = []
e_ls = []
bx_ls = []
for f in bfaces:
    bspline_f = convert_polygon2bspline_face(f)
    gfs = geomie3d.create.grids_frm_bspline_face(bspline_f, 1, 1)
    for gf in gfs:
        midpt = geomie3d.calculate.face_midxyz(gf)
        bbx = geomie3d.create.bboxes_frm_midpts([midpt], [v_size], [v_size], [v_size])[0]
        bbx_ls.append(bbx)
        bx = geomie3d.create.box(v_size, v_size, v_size, centre_pt = midpt)
        bx_ls .append(bx)
        bedges = geomie3d.get.edges_frm_solid(bx)
        e_ls.extend(bedges)

viz_dlist.append({'topo_list':bx_ls, 'colour': 'blue'})

ndir = 360
unitball = geomie3d.d4pispace.tgDirs(ndir)
#create the rays for each analyse pts
aly_vs = []
rays = []
v_ls = []
for dix in unitball.getDirList():
    dirx = [dix.x, dix.y, dix.z]
    vertex = geomie3d.create.vertex(dirx)
    v_ls.append(vertex)
    ray = geomie3d.create.ray(ray_orig, dirx)
    rays.append(ray)

# print('dirx', rays[6].dirx)
inter_res = geomie3d.calculate.rays_bboxes_intersect(rays, bbx_ls)

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
        
    # viz_dlist.append({'topo_list': vs, 'colour': 'green'})

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

# viz_dlist.append({'topo_list': dir_es, 'colour': 'green'})
geomie3d.viz.viz(viz_dlist)
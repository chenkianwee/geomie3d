import geomie3d
import geomie3d.viz

#----------------------------------------------------------------
#process the geometry point clouds
#read the pts file
#----------------------------------------------------------------
pts_path = ''
res_path = ''

viz_dlist = []
with open(pts_path) as f:
    lines = f.readlines()

xyzs_ls = []   
nx_ls = [] 
for l in lines:
    l = l.split(',')
    xyz = l[0:3]
    xyz = list(map(float, xyz))
    nx = l[3:6]
    nx = list(map(float, nx))
    xyzs_ls.append(xyz)
    nx_ls.append(nx)

orig_v = geomie3d.create.vertex_list(xyzs_ls)

rot_mat = geomie3d.calculate.rotate_matrice([1,0,0], 90)
trsf_xyzs = geomie3d.calculate.trsf_xyzs(xyzs_ls, rot_mat)
trsf_nxs = geomie3d.calculate.trsf_xyzs(nx_ls, rot_mat)
print(nx_ls[0])
print(trsf_nxs[0])
bbx = geomie3d.calculate.bbox_frm_xyzs(trsf_xyzs)
midpt = geomie3d.calculate.bbox_centre(bbx)
print(midpt)
trsl_mat = geomie3d.calculate.translate_matrice(0-midpt[0], 0-midpt[1], 0-bbx.minz)
trsf_xyzs2 = geomie3d.calculate.trsf_xyzs(trsf_xyzs, trsl_mat)

trsf_vs = []
for cnt,trsf_xyz in enumerate(trsf_xyzs2):
    trsf_v = geomie3d.create.vertex(trsf_xyz, attributes = {'normal': trsf_nxs[cnt]})
    trsf_vs.append(trsf_v)

geomie3d.utility.write2pts(trsf_vs, res_path)

viz_dlist.append({'topo_list':orig_v, 'colour':[0,0,1,0.3]})
viz_dlist.append({'topo_list':trsf_vs, 'colour':[1,0,0,0.6]})
geomie3d.viz.viz(viz_dlist)

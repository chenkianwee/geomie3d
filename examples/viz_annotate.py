import geomie3d
import geomie3d.viz

def create_scene():
    box = geomie3d.create.box(4.1, 3.8, 2.4)
    mv_box = geomie3d.modify.move_topo(box, [0,0,0], ref_xyz = [0,0,-1.2])
    srfs = geomie3d.get.faces_frm_solid(mv_box)
    bdry_srfs = []
    for cnt,s in enumerate(srfs):
        s = geomie3d.modify.reverse_face_normal(s)
        geomie3d.modify.update_topo_att(s, {'name': 'surface' + str(cnt), 'count': cnt})
        bdry_srfs.append(s)
    return bdry_srfs

bdry_srfs = create_scene()

geomie3d.viz.viz([{'topo_list': bdry_srfs, 'colour': 'blue', 'attribute': 'name'}])
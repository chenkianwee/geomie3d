import geomie3d
import geomie3d.viz

bbox = geomie3d.utility.Bbox([1,1,5,10,10,10])
div_bboxes = geomie3d.create.grid3d_from_bbox(bbox, 5, 5, 5)

big_box = geomie3d.create.boxes_frm_bboxes([bbox])
boxes = geomie3d.create.boxes_frm_bboxes(div_bboxes)

# geomie3d.viz.viz([{'topo_list': boxes, 'colour': 'blue'},
#                   {'topo_list': big_box, 'colour': 'red'}])

geomie3d.viz.viz([{'topo_list': boxes, 'colour': 'blue'}])
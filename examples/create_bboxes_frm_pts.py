import geomie3d
import geomie3d.viz

import numpy as np

midpts = np.array([[0,0,0], [5,5,5]])
xdims = np.array([5,5])
ydims = np.array([5,5])
zdims = np.array([5,5])
att_ls = [{'id': 0}, {'id': 1}]
bboxes = geomie3d.create.bboxes_frm_midpts(midpts, xdims, ydims, zdims, attributes_list=att_ls)
boxes = geomie3d.create.boxes_frm_bboxes(bboxes)
# geomie3d.viz.viz([{'topo_list': boxes, 'colour': 'blue'}])

lwr_left_pts = np.array([[0,0,0], [5,5,5]])
xdims = np.array([5,5])
ydims = np.array([5,5])
zdims = np.array([0,0])
att_ls = [{'id': 0}, {'id': 1}]
bboxes = geomie3d.create.bboxes_frm_lwr_left_pts(lwr_left_pts, xdims, ydims, zdims, attributes_list=att_ls)
boxes1 = geomie3d.create.boxes_frm_bboxes(bboxes)

geomie3d.viz.viz([{'topo_list': boxes1, 'colour': 'blue'},
                  {'topo_list': boxes, 'colour': 'red'}])
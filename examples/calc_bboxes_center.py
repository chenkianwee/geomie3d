import geomie3d

bbox1 = geomie3d.create.bbox_frm_arr([1,1,0,10,10,5])
bbox2 = geomie3d.create.bbox_frm_arr([2,2,1,8,8,4])
center_pts = geomie3d.calculate.bboxes_centre([bbox1, bbox2])
print(center_pts)
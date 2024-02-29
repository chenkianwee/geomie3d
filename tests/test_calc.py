import geomie3d

def are_bbox1_related_bbox2():
    bbox1 = geomie3d.utility.Bbox([1,1,0,10,10,5])
    bbox2 = geomie3d.utility.Bbox([1,1,0,10,10,5])

    bbox3 = geomie3d.utility.Bbox([2,2,1,8,8,4])
    bbox4 = geomie3d.utility.Bbox([2,2,1,8,8,4])

    bbox5 = geomie3d.utility.Bbox([12,12,11,18,18,14])
    bbox6 = geomie3d.utility.Bbox([15,19,11,21,23,14])

    are_related = geomie3d.calculate.are_bboxes1_related2_bboxes2([bbox1, bbox2, bbox5], [bbox3, bbox4, bbox6])
    print(are_related)

def bbox_center():
    bbox1 = geomie3d.create.bbox_frm_arr([1,1,0,10,10,5])
    bbox2 = geomie3d.create.bbox_frm_arr([2,2,1,8,8,4])
    center_pts = geomie3d.calculate.bboxes_centre([bbox1, bbox2])
    print(center_pts)
import geomie3d

bbox1 = geomie3d.utility.Bbox([1,1,0,10,10,5])
bbox2 = geomie3d.utility.Bbox([1,1,0,10,10,5])

bbox3 = geomie3d.utility.Bbox([0,2,1,8,8,4])
bbox4 = geomie3d.utility.Bbox([2,2,1,8,8,4])

bbox5 = geomie3d.utility.Bbox([12,12,11,18,18,14])
bbox6 = geomie3d.utility.Bbox([15,19,11,21,23,14])

are_contained = geomie3d.calculate.are_bboxes1_in_bboxes2([bbox3, bbox4, bbox5], [bbox1, bbox2, bbox6])
print(are_contained)
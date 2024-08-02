import geomie3d
import geomie3d.viz
from dateutil.parser import parse

def test_viz_animate_falsecolor():
    # the geometry data to view
    ctrl_pts = [[1,5,0], [5,5,0],
                [1,0,0], [5,0,0]]

    deg_u = 1
    deg_v = 1

    kv_u = 2
    kv_v = 2

    f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v)
    g = geomie3d.create.grids_frm_bspline_face(f, 5, 5)

    bx = geomie3d.create.box(5, 5, 5)
    edges1 = geomie3d.get.edges_frm_solid(bx)
    bx2 = geomie3d.create.box(10, 10, 5)
    edges2 = geomie3d.get.edges_frm_solid(bx2)
    bx3 = geomie3d.create.box(15, 15, 10)
    edges3 = geomie3d.get.edges_frm_solid(bx3)

    res = []
    for i in range(len(g)):
        res.append(i*10)

    res2 = []
    for i in range(len(g)):
        res2.append(i*2)

    res3 = []
    for i in range(len(g)):
        res3.append(i*5)

    topo2d = [g, g, g]
    res2d = [res, res2, res3]
    topo_datetime_ls = [parse('2023-02-15T13:51'), parse('2023-02-15T12:51'), parse('2023-02-15T14:51')]
    topo_2ddlist = [[{'topo_list': edges1, 'colour': 'red'}], 
                    [{'topo_list': edges2, 'colour': 'green'}], 
                    [{'topo_list': edges3, 'colour': 'blue'}]]

    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = dt.timestamp()
        unix43d.append(unix_time)

    geomie3d.viz.pg.mkQApp()
    win = geomie3d.viz.AFalsecolour(topo2d, res2d, unix43d, false_min_max_val = [0,240], 
                                    other_topo_2ddlist = topo_2ddlist)
    
    win.setWindowTitle("AnimateFalseColour")
    win.show()
    win.resize(1100,700)

def test_viz_animate_topo():
    # the geometry data to view
    ctrl_pts = [[1,5,0], [5,5,0],
                [1,0,0], [5,0,0]]

    deg_u = 1
    deg_v = 1

    kv_u = 2
    kv_v = 2

    f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v)
    g = geomie3d.create.grids_frm_bspline_face(f, 5, 5)

    bx = geomie3d.create.box(5, 5, 5)
    edges1 = geomie3d.get.edges_frm_solid(bx)
    bx2 = geomie3d.create.box(10, 10, 5)
    edges2 = geomie3d.get.edges_frm_solid(bx2)
    bx3 = geomie3d.create.box(15, 15, 10)
    edges3 = geomie3d.get.edges_frm_solid(bx3)
    topo_datetime_ls = [parse('2023-02-15T13:51'), parse('2023-02-15T12:51'), parse('2023-02-15T14:51')]

    topo_2ddlist = [[{'topo_list': edges1, 'colour': 'red'}, {'topo_list': g, 'colour': 'blue'}], 
                    [{'topo_list': edges2, 'colour': 'green'}, {'topo_list': g, 'colour': 'green'}], 
                    [{'topo_list': edges3, 'colour': 'blue'}, {'topo_list': g, 'colour': 'blue'}]]
    
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = dt.timestamp()
        unix43d.append(unix_time)

    geomie3d.viz.pg.mkQApp()
    win = geomie3d.viz.AnimateTopo(topo_2ddlist, unix43d)
    
    win.setWindowTitle("AnimateTopo")
    win.show()
    win.resize(1100,700)

def test_viz_annotate():
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
    viz_dict = {'topo_list': bdry_srfs, 'colour': 'blue', 'attribute': 'name'}
    geomie3d.viz.pg.mkQApp()
    win = geomie3d.viz.VizTopo([viz_dict], gl_option='opaque')
    win.setWindowTitle("Viz")
    win.show()
    win.resize(1100,700)

def test_viz_falsecolor():
    pass

def test_viz_graph():
    pass

def test_viz_spatial_time_series():
    # the geometry data to view
    ctrl_pts = [[1,5,0], [5,5,0],
                [1,0,0], [5,0,0]]

    deg_u = 1
    deg_v = 1

    kv_u = 2
    kv_v = 2

    f = geomie3d.create.bspline_face_frm_ctrlpts(ctrl_pts, kv_u, kv_v, deg_u, deg_v)
    g = geomie3d.create.grids_frm_bspline_face(f, 5, 5)

    bx = geomie3d.create.box(5, 5, 5)
    edges1 = geomie3d.get.edges_frm_solid(bx)
    bx2 = geomie3d.create.box(10, 10, 5)
    edges2 = geomie3d.get.edges_frm_solid(bx2)
    bx3 = geomie3d.create.box(15, 15, 10)
    edges3 = geomie3d.get.edges_frm_solid(bx3)

    res = []
    for i in range(len(g)):
        res.append(i*10)

    res2 = []
    for i in range(len(g)):
        res2.append(i*2)

    res3 = []
    for i in range(len(g)):
        res3.append(i*5)

    topo2d = [g, g, g]
    res2d = [res, res2, res3]
    topo_datetime_ls = [parse('2023-02-15T13:51'), parse('2023-02-15T12:51'), parse('2023-02-15T14:51')]
    topo_2ddlist = [[{'topo_list': edges1, 'colour': 'red'}], 
                    [{'topo_list': edges2, 'colour': 'green'}], 
                    [{'topo_list': edges3, 'colour': 'blue'}]]

    #the time-series data
    dates_str2d = [['2023-02-15T13:51-0400',
                    '2023-02-15T12:51',
                    '2023-02-15T11:31',
                    '2023-02-15T10:11',
                    '2023-02-15T09:02'],
                ['2023-02-16T10:51',
                    '2023-02-16T09:51',
                    '2023-02-16T08:31',
                    '2023-02-16T07:11',
                    '2023-02-16T06:02']]

    yvalues2d = [[10, 14, 18, 20, 26], 
                [10, 15, 35, 6, 28]]

    #parse the string to datetime
    dates2d = []
    for dates_str in dates_str2d:
        dates = []
        for d in dates_str:
            dates.append(parse(d))
        
        dates2d.append(dates)
        
    colour_ls = [[255,0,0,255], [0,255,0,255]]

    infl_dicts = [{'label': 'linex', 'angle': 90, 'pos': parse('2023-02-16T06:02'), 'colour': (255,255,255,255)},
                {'label': 'liney', 'angle': 0, 'pos': 20, 'colour': (255,255,255,255)}]


    region_dicts = [{'label': 'regionx', 'orientation': 'vertical', 'range': [parse('2023-02-16T05:02'), parse('2023-02-16T08:02')], 'colour': [255,255,255,80]},
                    {'label': 'regiony', 'orientation': 'horizontal', 'range': [5,10], 'colour': [255,0,0,80]}]

    dates_str2d = [['2023-02-15T13:30',
                    '2023-02-15T12:30',
                    '2023-02-15T11:50',
                    '2023-02-15T10:00',
                    '2023-02-15T09:30'],
                ['2023-02-16T10:10',
                    '2023-02-16T09:20',
                    '2023-02-16T08:15',
                    '2023-02-16T07:00',
                    '2023-02-16T06:15']]

    #parse the string to datetime
    second_xvalues2d = []
    for dates_str in dates_str2d:
        dates = []
        for d in dates_str:
            dates.append(parse(d))
        second_xvalues2d.append(dates)
        
    second_yvalues2d = [[8,15,21,35,6],
                        [1,28,14,9,3]]

    second_colour_ls = [[255,255,0,150], [0,255,255,150]]

    #convert the datetimes to unix timestamp
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = dt.timestamp()
        unix43d.append(unix_time)
    
    unix_2d_graph = geomie3d.viz._convert_datetime2unix(dates2d)
    
    if second_xvalues2d is not None:
        second_xvalues2d  = geomie3d.viz._convert_datetime2unix(second_xvalues2d)
        
    geomie3d.viz.pg.mkQApp()
    win = geomie3d.viz.StView(topo2d, res2d, unix43d, unix_2d_graph, yvalues2d, colour_ls, false_min_max_val=[0,240],
                              other_topo_2ddlist=topo_2ddlist, xlabel = 'time', xunit = None, ylabel = 'Something', yunit='someunit', 
                              title = 'Some Title', legend = ['someThingX', 'thatThingY'], inf_lines = infl_dicts, regions = region_dicts, 
                              second_xvalues2d=second_xvalues2d, second_yvalues2d=second_yvalues2d, second_colour_ls=second_colour_ls, 
                              second_legend=['someThingA', 'someThingB'], second_ylabel='something2', second_yunit='someunit2')
    
    global p1, p2
    win.setWindowTitle("SpatialTimeSeriesView")
    win.showMaximized()

def test_viz_topo():
    #defining the boundary wire
    xyz_list1 = [(10,10,0), (20,10,0), (20,20,0)]
    att_list1 = [{"type":"bus stop"}, {"type":"shop"}, {"type":"monument"}]
    vlist1 = geomie3d.create.vertex_list(xyz_list1, attributes_list = att_list1)
    pl_att1 = {"id":0, "type": "sidewalk"}
    pline_edge1 = geomie3d.create.pline_edge_frm_verts(vlist1, attributes = pl_att1)

    xyz_list2 = [(20,20,0), (10,20,0), (10,10,0)]
    att_list2 = [{"type":{"test":1}}, {"type":"shop"}, {"type":"bus stop", "id":100}]
    vlist2 = geomie3d.create.vertex_list(xyz_list2, attributes_list = att_list2)
    pl_att2 = {"id":1, "type": "road"}
    pline_edge2 = geomie3d.create.pline_edge_frm_verts(vlist2, attributes = pl_att2)

    bdry_edge_list = [pline_edge1, pline_edge2]
    w_att1 = {"id":0, "type": "plot"}
    bdry_wire = geomie3d.create.wire_frm_edges(bdry_edge_list, attributes = w_att1)

    #defining the hole wire
    xyz_list3 = [(12,12,0), (18,12,0), (18,18,0)]
    vlist3 = geomie3d.create.vertex_list(xyz_list3)
    pline_edge3 = geomie3d.create.pline_edge_frm_verts(vlist3)

    xyz_list4 = [(18,18,0), (12,18,0), (12,12,0)]
    vlist4 = geomie3d.create.vertex_list(xyz_list4)
    pline_edge4 = geomie3d.create.pline_edge_frm_verts(vlist4)

    hole_edge_list = [pline_edge3, pline_edge4]
    hole_wire = geomie3d.create.wire_frm_edges(hole_edge_list)

    #create the face
    f_att = {"type": "residential", "id": 0}
    face = geomie3d.create.polygon_face_frm_wires(bdry_wire, [hole_wire], attributes = f_att)

    viz_dict = {"topo_list":[face], "colour": [0,1,0,1], "attributes": []}
    geomie3d.viz.pg.mkQApp()
    win = geomie3d.viz.VizTopo([viz_dict], gl_option='opaque')
    win.setWindowTitle("Viz")
    win.show()
    win.resize(1100,700)
import geomie3d
import geomie3d.viz
from dateutil.parser import parse

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

geomie3d.viz.viz_st(topo2d, res2d, topo_datetime_ls, dates2d, yvalues2d, colour_ls, false_min_max_val=[0,240],
                    other_topo_2ddlist=topo_2ddlist, xlabel = 'time', xunit = None, ylabel = 'Something', yunit='someunit', 
                    title = 'Some Title', legend = ['someThingX', 'thatThingY'], inf_lines = infl_dicts, regions = region_dicts, 
                    second_xvalues2d=second_xvalues2d, second_yvalues2d=second_yvalues2d, second_colour_ls=second_colour_ls, 
                    second_legend=['someThingA', 'someThingB'], second_ylabel='something2', second_yunit='someunit2')

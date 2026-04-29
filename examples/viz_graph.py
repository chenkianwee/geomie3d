import geomie3d
import geomie3d.viz
from dateutil.parser import parse

dates_str2d = [['2023-02-15T13:51',
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

infl_dicts = [{'label': 'linex', 'angle': 90, 'pos': parse('2023-02-16T06:02'), 'colour': [0,0,255,255]},
              {'label': 'liney', 'angle': 0, 'pos': 20, 'colour': (255,255,255,255)}]

region_dicts = [{'label': 'regionx', 'orientation': 'vertical', 'range': [parse('2023-02-16T05:02'), parse('2023-02-16T08:02')], 'colour': [255,255,255,80]},
                {'label': 'regiony', 'orientation': 'horizontal', 'range': [5,10], 'colour': [255,0,0,80]}]

colour_ls = [[255,0,0,255], [0,255,0,255]]
geomie3d.viz.viz_graph(dates2d, yvalues2d, colour_ls, xlabel='Time', ylabel='Some value', yunit='someunit', title='Example Graph',
                       legend = ['data1', 'data2'], inf_lines=infl_dicts, regions=region_dicts, second_xvalues2d=second_xvalues2d,
                       second_yvalues2d=second_yvalues2d, second_colour_ls=second_colour_ls, second_ylabel='axis2', 
                       second_yunit='axis2unit', second_legend = ['dataa', 'datab'])

xvalues = [[1,2,3,4,5], [2,5,8,10,12]]
geomie3d.viz.viz_graph(xvalues, yvalues2d, colour_ls, xlabel='xvalue', xunit = 'xunit', ylabel='Some value', yunit='someunit', 
                       title='Example Graph', legend = ['datax', 'datay'])
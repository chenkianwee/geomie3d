# ==================================================================================================
#
#    Copyright (c) 2020, Chen Kian Wee (chenkianwee@gmail.com)
#
#    This file is part of geomie3d
#
#    geomie3d is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geomie3d is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with py4design.  If not, see <http://www.gnu.org/licenses/>.
#
# ==================================================================================================
import os
import sys
import time
import datetime
from dateutil.parser import parse
from itertools import chain

import numpy as np

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
import pyqtgraph.opengl as gl

from . import modify
from . import get
from . import create
from . import calculate
from . import topobj

#setup the QT environment
# import PyQT6
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt6'
if sys.platform == 'linux' or sys.platform == 'linux2':
    os.environ['QT_QPA_PLATFORM'] = 'wayland-egl'
    # os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
def viz(topo_dictionary_list):
    """
    This function visualises the topologies.
 
    Parameters
    ----------
    topo_dictionary_list : a list of dictionary
        A list of dictionary specifying the visualisation parameters.
        topo_list: the topos to visualise
        colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        draw_edges: bool whether to draw the edges of mesh, default is True
        point_size: size of the point
        px_mode: True or False, if true the size of the point is in pixel if not its in meters
    """    
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    pg.mkQApp('')
    w = gl.GLViewWidget()
    w.clear()
    bbox_list = _convert_topo_dictionary_list4viz(topo_dictionary_list, w)
    
    w.addItem(gl.GLAxisItem())
    overall_bbox = calculate.bbox_frm_bboxes(bbox_list)
    midpt = calculate.bbox_centre(overall_bbox)
    w.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
    
    lwr_left = [overall_bbox.minx, overall_bbox.miny, overall_bbox.minz]
    upr_right =  [overall_bbox.maxx, overall_bbox.maxy, overall_bbox.maxz]
    dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
    w.opts['distance'] = dist*1.5
    # w.setBackgroundColor('w')
    w.show()
    w.setWindowTitle('Geomie3D viz')
    pg.exec()

def viz_falsecolour(topo_list, results, false_min_max_val = None, other_topo_dlist = []):
    """
    This function visualises the topologies in falsecolour.
 
    Parameters
    ----------
    topo_list : list of topologies
        list of topologies.
    
    results : list of floats
        list of performance value to be visualise as falsecolour.
    
    min_max_val : list of floats, optional
        list of two values [min_val, max_val] for the falsecolour visualisation.
    
    other_topo_dlist : a list of dictionary, optional
        A list of dictionary specifying the visualisation parameters.
        topo_list: the topos to visualise
        colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        draw_edges: bool whether to draw the edges of mesh, default is True
        point_size: size of the point
        px_mode: True or False, if true the size of the point is in pixel if not its in meters
    
    """
    class FalsecolourView(QtWidgets.QWidget):
        def __init__(self):
            QtWidgets.QWidget.__init__(self)
            self.setupGUI()
            
        def setupGUI(self):
            self.layout = QtWidgets.QVBoxLayout()
            self.layout.setContentsMargins(0,0,0,0)
            self.setLayout(self.layout)
            #create the right panel
            self.splitter = QtWidgets.QSplitter()
            self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
            #create the parameter tree for housing the parameters
            self.tree = ParameterTree(showHeader=False)
            self.splitter.addWidget(self.tree)
            #put the splitter 2 into the layout
            self.layout.addWidget(self.splitter)
            
            self.view3d = gl.GLViewWidget()
            self.splitter.addWidget(self.view3d)
        
        def insert_fcolour(self, fmin_val, fmax_val, results):
            self.min_val = min(results)
            self.max_val = max(results)
            falsecolour, int_ls, clr_ls = self.gen_falsecolour_bar(fmin_val, fmax_val)
            self.falsecolour = falsecolour
            
            self.min_max = dict(name='Min Max', type='group', expanded = True, title = "Min Max Value",
                                children=[dict(name='Min Value', type = 'float', title = "Min Value", value = self.min_val, readonly = True),
                                          dict(name='Max Value', type = 'float', title = "Max Value", value = self.max_val, readonly = True)]
                                )
            self.params = Parameter.create(name = "Parmx", type = "group", children = [self.falsecolour,
                                                                                         self.min_max])
            
            self.tree.setParameters(self.params, showTop=False)
            return int_ls, clr_ls
        
        def gen_falsecolour_bar(self, min_val, max_val):
            interval = 10.0
            inc1 = (max_val-min_val)/(interval)
            inc2 = inc1/2.0    
            float_list = list(np.arange(min_val+inc2, max_val, inc1))
            
            rangex = max_val-min_val
            intervals = rangex/10.0
            intervals_half = intervals/2.0
            
            interval_ls = []
            str_list = []
            for fcnt,f in enumerate(float_list):
                mi = round(f - intervals_half, 2)
                ma = round(f + intervals_half, 2)
                if fcnt == 0:
                    strx = "<" + str(ma)
                elif fcnt == 9:
                    strx = ">" + str(mi)
                else:
                    strx = str(mi) + " - " + str(ma)
                    
                str_list.append(strx)
                interval_ls.append([mi,ma])
                
            # bcolour = calc_falsecolour(float_list, min_val, max_val)
            bcolour = [[0,0,0.5], [0,1,1], [0,0.5,0], [0,1,0], [1,1,0], 
                       [1,0.59,0], [1,0,1], [0.42,0.33,0.33], [0.5,0,0], [1,0,0]]
            
            new_c_list = []
            for c in bcolour:
                new_c = [c[0]*255, c[1]*255, c[2]*255]
                new_c_list.append(new_c)
            
            falsecolourd = dict(name='Falsecolour', type='group', expanded = True, title = "Colour Legend",
                                    children =  [dict(name = str_list[9], type = 'color', value = new_c_list[9], readonly = True),
                                                 dict(name = str_list[8], type = 'color', value = new_c_list[8], readonly = True),
                                                 dict(name = str_list[7], type = 'color', value = new_c_list[7], readonly = True),
                                                 dict(name = str_list[6], type = 'color', value = new_c_list[6], readonly = True),
                                                 dict(name = str_list[5], type = 'color', value = new_c_list[5], readonly = True),
                                                 dict(name = str_list[4], type = 'color', value = new_c_list[4], readonly = True),
                                                 dict(name = str_list[3], type = 'color', value = new_c_list[3], readonly = True),
                                                 dict(name = str_list[2], type = 'color', value = new_c_list[2], readonly = True),
                                                 dict(name = str_list[1], type = 'color', value = new_c_list[1], readonly = True),
                                                 dict(name = str_list[0], type = 'color', value = new_c_list[0], readonly = True)]
                                )
            return falsecolourd, interval_ls, bcolour
    #========================================================================
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    pg.mkQApp()
    win = FalsecolourView()
    win.setWindowTitle("FalseColourView")
    if false_min_max_val == None:
        int_ls, c_ls = win.insert_fcolour(min(results), max(results), results)
    else:
        int_ls, c_ls = win.insert_fcolour(false_min_max_val[0], 
                                          false_min_max_val[1], results)

    topo_int_ls = _create_topo_int_ls(int_ls)
    # print(len(topo_list))
    #sort all the topologies into the 10 intervals
    for cnt,topo in enumerate(topo_list):
        res = results[cnt]
        topo_type = topo.topo_type
        for icnt, intx in enumerate(int_ls):
            if icnt == 0:
                if res < intx[1]:
                    _sort_topos(topo, topo_type, icnt, topo_int_ls)
                    break
                
            elif icnt == len(int_ls)-1:
                if res >= intx[0]:
                    _sort_topos(topo, topo_type, icnt, topo_int_ls)
                    break
                
            else:
                if intx[0] <= res < intx[1]:
                    _sort_topos(topo, topo_type, icnt, topo_int_ls)
                    break
    
    #create visualisable topologies for visualisation
    for ccnt, topo_int in enumerate(topo_int_ls):
        rgb = list(c_ls[ccnt])
        rgb.append(1)
        #viz pts
        all_pts = topo_int[0]
        all_edges = topo_int[1]
        all_faces = topo_int[2]
        if len(all_pts) > 0:
            all_pts = np.array(all_pts)
            viz_pts = _make_points(all_pts, rgb, 10, pxMode = True)
            win.view3d.addItem(viz_pts)
        if len(all_edges) > 0:
            line_vertices = modify.edges2lines(all_edges)
            viz_lines = _make_line(line_vertices, line_colour = rgb)
            win.view3d.addItem(viz_lines)
        if len(all_faces) > 0:
            mesh_dict = modify.faces2mesh(all_faces)
            #flip the indices
            verts = mesh_dict['vertices']
            idx = mesh_dict['indices']
            #flip the vertices to be clockwise
            idx = np.flip(idx, axis=1)
            viz_mesh = _make_mesh(verts, idx, 
                                  draw_edges = False)
            viz_mesh.setColor(np.array(rgb))
            win.view3d.addItem(viz_mesh)
            
    win.view3d.addItem(gl.GLAxisItem())
    cmp = create.composite(topo_list)
    bbox = calculate.bbox_frm_topo(cmp)
    midpt = calculate.bbox_centre(bbox)
    win.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
    
    lwr_left = [bbox.minx, bbox.miny, bbox.minz]
    upr_right =  [bbox.maxx, bbox.maxy, bbox.maxz]
    dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
    win.view3d.opts['distance'] = dist*1.5
    if len(other_topo_dlist) != 0:
        _convert_topo_dictionary_list4viz(other_topo_dlist, win.view3d)
    
    win.show()
    win.resize(1100,700)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()

def viz_st(topo_2dlist, results2d, topo_datetime_ls, xvalues2d, yvalues2d, colour_ls, false_min_max_val = None, 
           other_topo_2ddlist = [], xlabel = None, xunit = None, ylabel = None, yunit = None, title = None, legend = None, 
           inf_lines = None, regions = None):
    """
    This function visualises the spatial time-series data.
 
    Parameters
    ----------
    topo_2dlist : 2dlist of topologies
        2dlist of topologies. You can have one set of geometries to multiple sets of results.
    
    results2d : 2dlist of floats
        2d list of performance value to be visualise as falsecolour.
    
    topo_datetime_ls : list of datetime
        timestamp of the topology data.
    
    xvalues2d : 2dlist of datetime
        xvalues to plot. For different set of data separate them into different list.
    
    yvalues2d : 2dlist of floats
        yvalues to plot. For different set of data separate them into different list.
    
    colour_ls : list of tuple
        list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)].
    
    false_min_max_val : list of floats, optional
        list of two values [min_val, max_val] for the falsecolour visualisation.
    
    other_topo_2ddlist : a 2d list of dictionary, optional
        A 2d list of dictionary specifying the visualisation parameters.
        topo_list: the topos to visualise
        colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        draw_edges: bool whether to draw the edges of mesh, default is True
        point_size: size of the point
        px_mode: True or False, if true the size of the point is in pixel if not its in meters
    
    xlabel : str, optional
        label for the x-axis.
    
    xunit : str, optional
        unit label for the x-axis.
        
    ylabel : str, optional
        label for the y-axis.
    
    yunit : str, optional
        unit label for the y-axis.
        
    title : str, optional
        title for the graph.
    
    legend : list of str, optional
        list of strings for the legends.
        
    inf_lines : list of dictionary, optional
        dictionary describing the infinite line.
        label: str to describe the infinite line
        angle: float, the angle of the infinite line, 0=horizontal, 90=vertical
        pos: float, for horizontal and vertical line a single value is sufficient, for slanted line (x,y) is required.
        colour: tuple, (r,g,b,a)
        
    regions : list of dictionary, optional
        dictionary describing the region on the graph.
        label: str to describe the region.
        orientation: str, vertical or horizontal
        range: list of floats, [lwr_limit, upr_limit] e.g. [50,70] 
        colour: tuple, (r,g,b,a)
    """
    class StView(QtWidgets.QWidget):
        def __init__(self, topo_2dlist, results2d, topo_unixtime, xvalues2d, yvalues2d, colour_ls, false_min_max_val = None, 
                     other_topo_2ddlist = [], xlabel = None, xunit = None, ylabel = None, yunit = None, title = None, 
                     legend = None, inf_lines = None, regions = None):
            QtWidgets.QWidget.__init__(self)
            self.setupGUI()
            self.topo_2dlist = topo_2dlist
            self.results2d = results2d
            self.topo_unixtime = topo_unixtime
            self.xvalues2d = xvalues2d
            self.yvalues2d = yvalues2d
            self.colour_ls = colour_ls
            self.false_min_max_val = false_min_max_val
            self.other_topo_2ddlist = other_topo_2ddlist
            self.xlabel = xlabel
            self.xunit = xunit
            self.ylabel = ylabel
            self.yunit = yunit
            self.title = title
            self.legend = legend
            self.inf_lines = inf_lines
            self.regions = regions
            
            #process data to visualise 3d model
            res_flat = list(chain(*results2d))
            if false_min_max_val is None:
                int_ls, c_ls = self.insert_fcolour(min(res_flat), max(res_flat), res_flat)
            else:
                int_ls, c_ls = self.insert_fcolour(false_min_max_val[0], 
                                                  false_min_max_val[1], res_flat)
            self.int_ls = int_ls
            self.c_ls = c_ls
            self.insert_parm3d()
            # check if there are multiple sets of geometry or just one set of geometry with multiple results
            self.create_3d_dict()
            self.topo_unixtime_sorted = sorted(self.model3d_dict.keys())
            self.current_time_index = 0
            self.n3dtimes = len(self.topo_unixtime_sorted)
            
            start_3d_unixt = self.topo_unixtime_sorted[0]
            end_3d_unixt = self.topo_unixtime_sorted[-1]
            date_str_start = datetime.datetime.fromtimestamp(start_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
            date_str_end = datetime.datetime.fromtimestamp(end_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
            range_str = date_str_start + ' to ' + date_str_end
            self.params2.param('Parm3d').param('Data Range Loaded').setValue(range_str)
            #load the initial spatial time-series data
            self.load_3dmodel()
            #load the time-series data
            self.load_time_series_data()
            
        def setupGUI(self):
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            #create the right panel
            self.splitter = QtWidgets.QSplitter()
            self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
            #create the parameter tree for housing the parameters
            self.tree = ParameterTree(showHeader=False)
            self.splitter.addWidget(self.tree)
            #create the 3d view
            self.view3d = gl.GLViewWidget()
            self.splitter.addWidget(self.view3d)
            #create a vertical split for the graph to be at the bottom
            self.splitter2 = QtWidgets.QSplitter()
            self.splitter2.setOrientation(QtCore.Qt.Orientation.Vertical)
            #setup the graph plot
            self.plot = pg.plot()
            self.plot.setAxisItems(axisItems = {'bottom': pg.DateAxisItem()})
            self.plot.showGrid(x=True, y=True)
            #put the graph into the second 
            self.splitter2.addWidget(self.splitter)
            self.splitter2.addWidget(self.plot)
            #put the splitter2 into the layout
            self.layout.addWidget(self.splitter2)
            
            self.screen_width = self.screen().size().width()
            self.screen_height = self.screen().size().height()
            
            self.splitter.setSizes([int(self.screen_width/3.5), self.screen_width])
            self.splitter2.setSizes([self.screen_height, int(self.screen_height/1.5)])
        
            self.playback_speed = 1000
            
        def insert_fcolour(self, fmin_val, fmax_val, results):
            self.min_val = min(results)
            self.max_val = max(results)
            falsecolour, int_ls, clr_ls = self.gen_falsecolour_bar(fmin_val, fmax_val)
            self.falsecolour = falsecolour
            
            self.min_max = dict(name='Min Max', type='group', expanded = True, title = "Min Max Value",
                                children=[dict(name='Min Value', type = 'float', title = "Min Value", value = self.min_val, readonly = True),
                                          dict(name='Max Value', type = 'float', title = "Max Value", value = self.max_val, readonly = True)]
                                )
            self.params = Parameter.create(name = "Parmx", type = "group", children = [self.falsecolour,
                                                                                       self.min_max])
            
            self.tree.setParameters(self.params, showTop=False)
            return int_ls, clr_ls
        
        def gen_falsecolour_bar(self, min_val, max_val):
            interval = 10.0
            inc1 = (max_val-min_val)/(interval)
            inc2 = inc1/2.0    
            float_list = list(np.arange(min_val+inc2, max_val, inc1))
            
            rangex = max_val-min_val
            intervals = rangex/10.0
            intervals_half = intervals/2.0
            
            interval_ls = []
            str_list = []
            for fcnt,f in enumerate(float_list):
                mi = round(f - intervals_half, 2)
                ma = round(f + intervals_half, 2)
                if fcnt == 0:
                    strx = "<" + str(ma)
                elif fcnt == 9:
                    strx = ">" + str(mi)
                else:
                    strx = str(mi) + " - " + str(ma)
                    
                str_list.append(strx)
                interval_ls.append([mi,ma])
                
            # bcolour = calc_falsecolour(float_list, min_val, max_val)
            bcolour = [[0,0,0.5], [0,1,1], [0,0.5,0], [0,1,0], [1,1,0], 
                       [1,0.59,0], [1,0,1], [0.42,0.33,0.33], [0.5,0,0], [1,0,0]]
            
            new_c_list = []
            for c in bcolour:
                new_c = [c[0]*255, c[1]*255, c[2]*255]
                new_c_list.append(new_c)
            
            falsecolourd = dict(name='Falsecolour', type='group', expanded = True, title = "Colour Legend",
                                    children =  [dict(name = str_list[9], type = 'color', value = new_c_list[9], readonly = True),
                                                 dict(name = str_list[8], type = 'color', value = new_c_list[8], readonly = True),
                                                 dict(name = str_list[7], type = 'color', value = new_c_list[7], readonly = True),
                                                 dict(name = str_list[6], type = 'color', value = new_c_list[6], readonly = True),
                                                 dict(name = str_list[5], type = 'color', value = new_c_list[5], readonly = True),
                                                 dict(name = str_list[4], type = 'color', value = new_c_list[4], readonly = True),
                                                 dict(name = str_list[3], type = 'color', value = new_c_list[3], readonly = True),
                                                 dict(name = str_list[2], type = 'color', value = new_c_list[2], readonly = True),
                                                 dict(name = str_list[1], type = 'color', value = new_c_list[1], readonly = True),
                                                 dict(name = str_list[0], type = 'color', value = new_c_list[0], readonly = True)]
                                )
            return falsecolourd, interval_ls, bcolour
        
        def insert_parm3d(self):
            self.parm3d = dict(name='Parm3d', type='group', expanded = True, title = "3D View Controls",
                       children=[
                                    dict(name='Data Range Loaded', type = 'str', title = "Data Range Loaded", readonly = True),
                                    dict(name='Current Date', type = 'str', title = "Current Date", readonly = True),
                                    dict(name='Search Date', type = 'str', title = "Search Date", value = "yyyy-mm-ddTHH:MM:SS"),
                                    dict(name='Search', type = 'action', title = "Search"),
                                    dict(name='Search Result', type = 'str', value = '', readonly = True),
                                    dict(name='Previous', type = 'action', title = "Previous"),
                                    dict(name='Next', type = 'action', title = "Next")
                                ]
                        )
            self.animate_parm = dict(name='Animate Parms', type='group', expanded = False, title = "Animate Parameters",
                       children=[
                                   dict(name='Play Status', type = 'str', title = "Play Status", value = 'Pause(Forward)', readonly=True),
                                   dict(name='Pause/Play', type = 'action', title = "Pause/Play"),
                                   dict(name='Rewind', type = 'action', title = "Rewind"),
                                   dict(name='Forward', type = 'action', title = "Forward"),
                                   dict(name='Seconds/Frame', type = 'float', title = "Seconds/Frame", value = 1.0),
                                   dict(name='Change Playback Speed', type = 'action', title = "Change Playback Speed")
                                ]
                        )
            self.params2 = Parameter.create(name = "View3dparm", type = "group", children = [self.parm3d, self.animate_parm])
            self.tree.addParameters(self.params2, showTop=False)
            self.params2.param('Parm3d').param("Search").sigActivated.connect(self.search)
            self.params2.param('Parm3d').param("Previous").sigActivated.connect(self.previous_scene)
            self.params2.param('Parm3d').param("Next").sigActivated.connect(self.next_scene)
            self.params2.param('Animate Parms').param("Pause/Play").sigActivated.connect(self.pause)
            self.params2.param('Animate Parms').param("Rewind").sigActivated.connect(self.rewind)
            self.params2.param('Animate Parms').param("Forward").sigActivated.connect(self.forward)
            self.params2.param('Animate Parms').param("Change Playback Speed").sigActivated.connect(self.change_speed)
        
        def load_3dmodel(self):
            #clear the 3dview
            self.clear_3dview()
            model3d_dict = self.model3d_dict
            int_ls = self.int_ls
            c_ls = self.c_ls
            topo_unixtime_sorted = self.topo_unixtime_sorted
            current_time_index = self.current_time_index
            
            current_unix_time = topo_unixtime_sorted[current_time_index]
            model_res = model3d_dict[current_unix_time] 
            topo_list = model_res[0]
            results = model_res[1]
            other_topo_dlist = model_res[2]
            
            topo_int_ls = _create_topo_int_ls(int_ls)
            # print(len(topo_list))
            #sort all the topologies into the 10 intervals
            for cnt,topo in enumerate(topo_list):
                res = results[cnt]
                topo_type = topo.topo_type
                for icnt, intx in enumerate(int_ls):
                    if icnt == 0:
                        if res < intx[1]:
                            _sort_topos(topo, topo_type, icnt, topo_int_ls)
                            break
                        
                    elif icnt == len(int_ls)-1:
                        if res >= intx[0]:
                            _sort_topos(topo, topo_type, icnt, topo_int_ls)
                            break
                        
                    else:
                        if intx[0] <= res < intx[1]:
                            _sort_topos(topo, topo_type, icnt, topo_int_ls)
                            break
            
            #create visualisable topologies for visualisation
            for ccnt, topo_int in enumerate(topo_int_ls):
                rgb = list(c_ls[ccnt])
                rgb.append(1)
                #viz pts
                all_pts = topo_int[0]
                all_edges = topo_int[1]
                all_faces = topo_int[2]
                if len(all_pts) > 0:
                    all_pts = np.array(all_pts)
                    viz_pts = _make_points(all_pts, rgb, 10, pxMode = True)
                    self.view3d.addItem(viz_pts)
                if len(all_edges) > 0:
                    line_vertices = modify.edges2lines(all_edges)
                    viz_lines = _make_line(line_vertices, line_colour = rgb)
                    self.view3d.addItem(viz_lines)
                if len(all_faces) > 0:
                    mesh_dict = modify.faces2mesh(all_faces)
                    #flip the indices
                    verts = mesh_dict['vertices']
                    idx = mesh_dict['indices']
                    #flip the vertices to be clockwise
                    idx = np.flip(idx, axis=1)
                    viz_mesh = _make_mesh(verts, idx, 
                                          draw_edges = False)
                    viz_mesh.setColor(np.array(rgb))
                    self.view3d.addItem(viz_mesh)
                    
            self.view3d.addItem(gl.GLAxisItem())
            cmp = create.composite(topo_list)
            bbox = calculate.bbox_frm_topo(cmp)
            midpt = calculate.bbox_centre(bbox)
            self.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
            
            
            lwr_left = [bbox.minx, bbox.miny, bbox.minz]
            upr_right =  [bbox.maxx, bbox.maxy, bbox.maxz]
            dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
            self.view3d.opts['distance'] = dist*1.5
            
            if len(other_topo_dlist) != 0:
                _convert_topo_dictionary_list4viz(other_topo_dlist, self.view3d)
                
            # date_str = datetime.datetime.utcfromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
            date_str = datetime.datetime.fromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
            
            self.params2.param('Parm3d').param('Current Date').setValue(date_str)
            
        def create_3d_dict(self):
            topo_2dlist = self.topo_2dlist
            self.only1geom = False
            if len(topo_2dlist) == 1:
                self.only1geom = True
                
            results2d = self.results2d
            topo_unixtime = self.topo_unixtime
            other_topo_2ddlist = self.other_topo_2ddlist
            
            model3d_dict = {}
            if self.only1geom:
                for cnt, utime in enumerate(topo_unixtime):
                    topo_ls = topo_2dlist[0]
                    results = results2d[cnt]
                    if len(other_topo_2ddlist) != 0:
                        other_topo_dlist = other_topo_2ddlist[cnt]
                    else: 
                        other_topo_dlist = []
                    model3d_dict[utime] = [topo_ls, results, other_topo_dlist]
            else:
                for cnt, utime in enumerate(topo_unixtime):
                    topo_ls = topo_2dlist[cnt]
                    results = results2d[cnt]
                    if len(other_topo_2ddlist) != 0:
                        other_topo_dlist = other_topo_2ddlist[cnt]
                    else: 
                        other_topo_dlist = []
                    model3d_dict[utime] = [topo_ls, results, other_topo_dlist]
                    
            self.model3d_dict = model3d_dict
        
        def search(self):
            #get the date the user is searching for
            search_date = self.params2.param('Parm3d').param("Search Date").value()
            search_datetime = parse(search_date)
            search_unix_time = time.mktime(search_datetime.timetuple())
            
            topo_unixtime_sorted = self.topo_unixtime_sorted
            diff_ls = []
            for cnt,ut in enumerate(topo_unixtime_sorted):
                diff = search_unix_time - ut
                diff = abs(diff)
                diff_ls.append(diff)
            
            mn = min(diff_ls)
            new_index = diff_ls.index(mn)
            self.current_time_index = new_index
            new_unixtime = self.topo_unixtime_sorted[new_index]
            self.infl.setValue(new_unixtime)
            self.load_3dmodel()
            
            res_str = 'The result is ' + str(round(mn/60,2)) + 'mins from the search'
            self.params2.param('Parm3d').param("Search Result").setValue(res_str)
            
        def previous_scene(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            if current_time_index == 0:
                new_index = n3dtimes-1
            else:
                new_index = current_time_index - 1
            
            self.current_time_index = new_index
            new_unixtime = self.topo_unixtime_sorted[new_index]
            self.infl.setValue(new_unixtime)
            self.load_3dmodel()
        
        def next_scene(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            if current_time_index == n3dtimes-1:
                new_index = 0
            else:
                new_index = current_time_index + 1
            
            self.current_time_index = new_index
            new_unixtime = self.topo_unixtime_sorted[new_index]
            self.infl.setValue(new_unixtime)
            self.load_3dmodel()
            
        def update(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            rewind_status = self.rewind_status
            
            if rewind_status == True:
                if current_time_index == 0:
                    new_index = n3dtimes-1
                else:
                    new_index = current_time_index - 1
                    
            else:
                if current_time_index == n3dtimes-1:
                    new_index = 0
                else:
                    new_index = current_time_index + 1
            
            self.current_time_index = new_index
            new_unixtime = self.topo_unixtime_sorted[new_index]
            self.infl.setValue(new_unixtime)
            self.load_3dmodel()
            
        def pause(self):
            cur_status = self.timer_status
            status_str = ""
            if cur_status:
                self.timer.stop()
                self.timer_status = False
                status_str += "Pause"
            else:
                self.timer.start(self.playback_speed)
                self.timer_status = True
                status_str += "Play"
            
            rewind_state = self.rewind_status
            if rewind_state :
                status_str += "(Rewind)"
                self.params2.param('Animate Parms').param('Play Status').setValue(status_str)
            else:
                status_str += "(Forward)"
                self.params2.param('Animate Parms').param('Play Status').setValue(status_str)
        
        def rewind(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
            
            self.rewind_status = True
            self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
        
        def forward(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
            
            self.rewind_status = False
            self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
        
        def change_speed(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
                
            seconds = self.params2.param('Animate Parms').param('Seconds/Frame').value()
            self.playback_speed = int(seconds*1000)
            
            rewind_state = self.rewind_status
            if rewind_state :
                self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
            else:
                self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
        
        def load_time_series_data(self):
            #load the time-series data
            self.scatterplot_ls = _plot_graph(self.xvalues2d, self.yvalues2d, self.colour_ls)
            legend_item = pg.LegendItem((80,60), offset=(70,20))
            legend_item.setParentItem(self.plot.graphicsItem())
            if self.legend is None:
                for cnt, scatter in enumerate(self.scatterplot_ls):
                    self.plot.addItem(scatter)
                    legend_item.addItem(scatter, 'legend' + str(cnt))
            else:
                for cnt, scatter in enumerate(self.scatterplot_ls):
                    self.plot.addItem(scatter)
                    legend_item.addItem(scatter, self.legend[cnt])
            
            #add an infinite line to the graph
            init_pos = self.topo_unixtime_sorted[0]
            self.infl = pg.InfiniteLine(movable=False, angle=90, label='3DModelTime', pos = init_pos, pen = (255,255,255,255),
                                        labelOpts={'position':0.9, 'color': (0,0,0), 'fill': (255,255,255,150), 
                                                   'movable': True})
            inf_lines = self.inf_lines
            if inf_lines is not None:
                for infd in inf_lines:
                    label = infd['label']
                    angle = infd['angle']
                    init_pos = infd['pos']
                    colour = infd['colour']
                    if type(init_pos) == datetime.datetime:
                        init_pos = time.mktime(init_pos.timetuple())
                        
                    inf_line = pg.InfiniteLine(movable=False, angle=angle, label=label, pos = init_pos, pen = colour,
                                           labelOpts={'position':0.9, 'color': (0,0,0), 'fill': (255,255,255,150), 
                                                      'movable': True})
                    self.plot.addItem(inf_line)
                    
            regions = self.regions
            if regions is not None:
                for reg in regions:
                    label = reg['label']
                    orient = reg['orientation']
                    rangex = reg['range']
                    colour = reg['colour']
                    if type(rangex[0]) == datetime.datetime:
                        rangex = [time.mktime(rangex[0].timetuple()),
                                  time.mktime(rangex[1].timetuple())]
                    
                    lr = pg.LinearRegionItem(values=rangex, orientation=orient, brush=colour, pen=[255,255,255,80], movable=False)
                    label = pg.InfLineLabel(lr.lines[1], label, position=0.9, rotateAxis=(1,0), anchor=(1, 1))
                    self.plot.addItem(lr)
                    
            self.plot.addItem(self.infl)
            self.plot.setTitle(self.title)
            self.plot.setLabel('bottom', text = self.xlabel, units = self.xunit)
            self.plot.setLabel('left', text = self.ylabel, units = self.yunit)
        
        def clear_3dview(self):
            view_3d = self.view3d
            all_items = view_3d.items
            nitems = len(all_items)
            while (nitems !=0):
                for i in all_items:
                    view_3d.removeItem(i)
                
                all_items = view_3d.items
                nitems = len(all_items)
        
        def animate(self):
            timer = QtCore.QTimer()
            self.timer = timer
            timer.timeout.connect(self.update)
            # timer.start(self.playback_speed)
            self.timer_status = False
            self.rewind_status = False
            self.playback_speed = 1000
            self.start()
                
        def start(self):
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                pg.exec()
                
    #========================================================================
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    #convert the datetimes to unix timestamp
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = time.mktime(dt.timetuple())
        unix43d.append(unix_time)
    
    unix_2d_graph = _convert_datetime2unix(xvalues2d)
    pg.mkQApp()
    win = StView(topo_2dlist, results2d, unix43d, unix_2d_graph, yvalues2d, colour_ls, false_min_max_val = false_min_max_val, 
                 other_topo_2ddlist = other_topo_2ddlist, xlabel = xlabel, xunit = xunit, ylabel = ylabel, yunit = yunit, 
                 title = title, legend = legend, inf_lines = inf_lines, regions = regions)
    
    win.setWindowTitle("SpatialTimeSeriesView")
    win.showMaximized()
    win.animate()

def viz_animate_falsecolour(topo_2dlist, results2d, topo_datetime_ls, false_min_max_val = None, 
                            other_topo_2ddlist = []):
    """
    This function produces a falsecolour animation window.
 
    Parameters
    ----------
    topo_2dlist : 2dlist of topologies
        2dlist of topologies. You can have one set of geometries to multiple sets of results.
    
    results2d : 2dlist of floats
        2d list of performance value to be visualise as falsecolour.
    
    topo_datetime_ls : list of datetime
        timestamp of the topology data.
    
    false_min_max_val : list of floats, optional
        list of two values [min_val, max_val] for the falsecolour visualisation.
    
    other_topo_2ddlist : a 2d list of dictionary, optional
        A 2d list of dictionary specifying the visualisation parameters.
        topo_list: the topos to visualise
        colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        draw_edges: bool whether to draw the edges of mesh, default is True
        point_size: size of the point
        px_mode: True or False, if true the size of the point is in pixel if not its in meters
    
    """
    class AFalsecolour(QtWidgets.QWidget):
        def __init__(self, topo_2dlist, results2d, topo_unixtime, false_min_max_val = None, other_topo_2ddlist = []):
            QtWidgets.QWidget.__init__(self)
            self.setupGUI()
            self.topo_2dlist = topo_2dlist
            self.results2d = results2d
            self.topo_unixtime = topo_unixtime
            self.false_min_max_val = false_min_max_val
            self.other_topo_2ddlist = other_topo_2ddlist
            
            #process data to visualise 3d model
            res_flat = list(chain(*results2d))
            if false_min_max_val is None:
                int_ls, c_ls = self.insert_fcolour(min(res_flat), max(res_flat), res_flat)
            else:
                int_ls, c_ls = self.insert_fcolour(false_min_max_val[0], 
                                                  false_min_max_val[1], res_flat)
            self.int_ls = int_ls
            self.c_ls = c_ls
            self.insert_parm3d()
            # check if there are multiple sets of geometry or just one set of geometry with multiple results
            self.create_3d_dict()
            self.topo_unixtime_sorted = sorted(self.model3d_dict.keys())
            self.current_time_index = 0
            self.n3dtimes = len(self.topo_unixtime_sorted)
            
            start_3d_unixt = self.topo_unixtime_sorted[0]
            end_3d_unixt = self.topo_unixtime_sorted[-1]
            date_str_start = datetime.datetime.fromtimestamp(start_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
            date_str_end = datetime.datetime.fromtimestamp(end_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
            range_str = date_str_start + ' to ' + date_str_end
            self.params2.param('Parm3d').param('Data Range Loaded').setValue(range_str)
            #load the initial spatial time-series data
            self.load_3dmodel()
            
        def setupGUI(self):
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            #create the right panel
            self.splitter = QtWidgets.QSplitter()
            self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
            #create the parameter tree for housing the parameters
            self.tree = ParameterTree(showHeader=False)
            self.splitter.addWidget(self.tree)
            #create the 3d view
            self.view3d = gl.GLViewWidget()
            self.splitter.addWidget(self.view3d)
            self.layout.addWidget(self.splitter)
            
            self.screen_width = self.screen().size().width()
            self.screen_height = self.screen().size().height()
            self.splitter.setSizes([int(self.screen_width/3.5), self.screen_width])
            
            self.playback_speed = 1000
            
        def insert_fcolour(self, fmin_val, fmax_val, results):
            self.min_val = min(results)
            self.max_val = max(results)
            falsecolour, int_ls, clr_ls = self.gen_falsecolour_bar(fmin_val, fmax_val)
            self.falsecolour = falsecolour
            
            self.min_max = dict(name='Min Max', type='group', expanded = True, title = "Min Max Value",
                                children=[dict(name='Min Value', type = 'float', title = "Min Value", value = self.min_val, readonly = True),
                                          dict(name='Max Value', type = 'float', title = "Max Value", value = self.max_val, readonly = True)]
                                )
            self.params = Parameter.create(name = "Parmx", type = "group", children = [self.falsecolour,
                                                                                       self.min_max])
            
            self.tree.setParameters(self.params, showTop=False)
            return int_ls, clr_ls
        
        def gen_falsecolour_bar(self, min_val, max_val):
            interval = 10.0
            inc1 = (max_val-min_val)/(interval)
            inc2 = inc1/2.0    
            float_list = list(np.arange(min_val+inc2, max_val, inc1))
            
            rangex = max_val-min_val
            intervals = rangex/10.0
            intervals_half = intervals/2.0
            
            interval_ls = []
            str_list = []
            for fcnt,f in enumerate(float_list):
                mi = round(f - intervals_half, 2)
                ma = round(f + intervals_half, 2)
                if fcnt == 0:
                    strx = "<" + str(ma)
                elif fcnt == 9:
                    strx = ">" + str(mi)
                else:
                    strx = str(mi) + " - " + str(ma)
                    
                str_list.append(strx)
                interval_ls.append([mi,ma])
                
            # bcolour = calc_falsecolour(float_list, min_val, max_val)
            bcolour = [[0,0,0.5], [0,1,1], [0,0.5,0], [0,1,0], [1,1,0], 
                       [1,0.59,0], [1,0,1], [0.42,0.33,0.33], [0.5,0,0], [1,0,0]]
            
            new_c_list = []
            for c in bcolour:
                new_c = [c[0]*255, c[1]*255, c[2]*255]
                new_c_list.append(new_c)
            
            falsecolourd = dict(name='Falsecolour', type='group', expanded = True, title = "Colour Legend",
                                    children =  [dict(name = str_list[9], type = 'color', value = new_c_list[9], readonly = True),
                                                 dict(name = str_list[8], type = 'color', value = new_c_list[8], readonly = True),
                                                 dict(name = str_list[7], type = 'color', value = new_c_list[7], readonly = True),
                                                 dict(name = str_list[6], type = 'color', value = new_c_list[6], readonly = True),
                                                 dict(name = str_list[5], type = 'color', value = new_c_list[5], readonly = True),
                                                 dict(name = str_list[4], type = 'color', value = new_c_list[4], readonly = True),
                                                 dict(name = str_list[3], type = 'color', value = new_c_list[3], readonly = True),
                                                 dict(name = str_list[2], type = 'color', value = new_c_list[2], readonly = True),
                                                 dict(name = str_list[1], type = 'color', value = new_c_list[1], readonly = True),
                                                 dict(name = str_list[0], type = 'color', value = new_c_list[0], readonly = True)]
                                )
            return falsecolourd, interval_ls, bcolour
        
        def insert_parm3d(self):
            self.parm3d = dict(name='Parm3d', type='group', expanded = True, title = "3D View Controls",
                       children=[
                                    dict(name='Data Range Loaded', type = 'str', title = "Data Range Loaded", readonly = True),
                                    dict(name='Current Date', type = 'str', title = "Current Date", readonly = True),
                                    dict(name='Search Date', type = 'str', title = "Search Date", value = "yyyy-mm-ddTHH:MM:SS"),
                                    dict(name='Search', type = 'action', title = "Search"),
                                    dict(name='Search Result', type = 'str', value = '', readonly = True),
                                    dict(name='Previous', type = 'action', title = "Previous"),
                                    dict(name='Next', type = 'action', title = "Next")
                                ]
                        )
            self.animate_parm = dict(name='Animate Parms', type='group', expanded = False, title = "Animate Parameters",
                       children=[
                                   dict(name='Play Status', type = 'str', title = "Play Status", value = 'Pause(Forward)', readonly=True),
                                   dict(name='Pause/Play', type = 'action', title = "Pause/Play"),
                                   dict(name='Rewind', type = 'action', title = "Rewind"),
                                   dict(name='Forward', type = 'action', title = "Forward"),
                                   dict(name='Seconds/Frame', type = 'float', title = "Seconds/Frame", value = 1.0),
                                   dict(name='Change Playback Speed', type = 'action', title = "Change Playback Speed")
                                ]
                        )
            self.params2 = Parameter.create(name = "View3dparm", type = "group", children = [self.parm3d, self.animate_parm])
            self.tree.addParameters(self.params2, showTop=False)
            self.params2.param('Parm3d').param("Search").sigActivated.connect(self.search)
            self.params2.param('Parm3d').param("Previous").sigActivated.connect(self.previous_scene)
            self.params2.param('Parm3d').param("Next").sigActivated.connect(self.next_scene)
            self.params2.param('Animate Parms').param("Pause/Play").sigActivated.connect(self.pause)
            self.params2.param('Animate Parms').param("Rewind").sigActivated.connect(self.rewind)
            self.params2.param('Animate Parms').param("Forward").sigActivated.connect(self.forward)
            self.params2.param('Animate Parms').param("Change Playback Speed").sigActivated.connect(self.change_speed)
        
        def load_3dmodel(self):
            #clear the 3dview
            self.clear_3dview()
            model3d_dict = self.model3d_dict
            int_ls = self.int_ls
            c_ls = self.c_ls
            topo_unixtime_sorted = self.topo_unixtime_sorted
            current_time_index = self.current_time_index
            
            current_unix_time = topo_unixtime_sorted[current_time_index]
            model_res = model3d_dict[current_unix_time] 
            topo_list = model_res[0]
            results = model_res[1]
            other_topo_dlist = model_res[2]
            
            topo_int_ls = _create_topo_int_ls(int_ls)
            # print(len(topo_list))
            #sort all the topologies into the 10 intervals
            for cnt,topo in enumerate(topo_list):
                res = results[cnt]
                topo_type = topo.topo_type
                for icnt, intx in enumerate(int_ls):
                    if icnt == 0:
                        if res < intx[1]:
                            _sort_topos(topo, topo_type, icnt, topo_int_ls)
                            break
                        
                    elif icnt == len(int_ls)-1:
                        if res >= intx[0]:
                            _sort_topos(topo, topo_type, icnt, topo_int_ls)
                            break
                        
                    else:
                        if intx[0] <= res < intx[1]:
                            _sort_topos(topo, topo_type, icnt, topo_int_ls)
                            break
            
            #create visualisable topologies for visualisation
            for ccnt, topo_int in enumerate(topo_int_ls):
                rgb = list(c_ls[ccnt])
                rgb.append(1)
                #viz pts
                all_pts = topo_int[0]
                all_edges = topo_int[1]
                all_faces = topo_int[2]
                if len(all_pts) > 0:
                    all_pts = np.array(all_pts)
                    viz_pts = _make_points(all_pts, rgb, 10, pxMode = True)
                    self.view3d.addItem(viz_pts)
                if len(all_edges) > 0:
                    line_vertices = modify.edges2lines(all_edges)
                    viz_lines = _make_line(line_vertices, line_colour = rgb)
                    self.view3d.addItem(viz_lines)
                if len(all_faces) > 0:
                    mesh_dict = modify.faces2mesh(all_faces)
                    #flip the indices
                    verts = mesh_dict['vertices']
                    idx = mesh_dict['indices']
                    #flip the vertices to be clockwise
                    idx = np.flip(idx, axis=1)
                    viz_mesh = _make_mesh(verts, idx, 
                                          draw_edges = False)
                    viz_mesh.setColor(np.array(rgb))
                    self.view3d.addItem(viz_mesh)
                    
            self.view3d.addItem(gl.GLAxisItem())
            cmp = create.composite(topo_list)
            bbox = calculate.bbox_frm_topo(cmp)
            midpt = calculate.bbox_centre(bbox)
            self.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
            
            
            lwr_left = [bbox.minx, bbox.miny, bbox.minz]
            upr_right =  [bbox.maxx, bbox.maxy, bbox.maxz]
            dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
            self.view3d.opts['distance'] = dist*1.5
            
            if len(other_topo_dlist) != 0:
                _convert_topo_dictionary_list4viz(other_topo_dlist, self.view3d)
                
            # date_str = datetime.datetime.utcfromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
            date_str = datetime.datetime.fromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
            
            self.params2.param('Parm3d').param('Current Date').setValue(date_str)
            
        def create_3d_dict(self):
            topo_2dlist = self.topo_2dlist
            self.only1geom = False
            if len(topo_2dlist) == 1:
                self.only1geom = True
                
            results2d = self.results2d
            topo_unixtime = self.topo_unixtime
            other_topo_2ddlist = self.other_topo_2ddlist
            
            model3d_dict = {}
            if self.only1geom:
                for cnt, utime in enumerate(topo_unixtime):
                    topo_ls = topo_2dlist[0]
                    results = results2d[cnt]
                    if len(other_topo_2ddlist) != 0:
                        other_topo_dlist = other_topo_2ddlist[cnt]
                    else: 
                        other_topo_dlist = []
                    model3d_dict[utime] = [topo_ls, results, other_topo_dlist]
            else:
                for cnt, utime in enumerate(topo_unixtime):
                    topo_ls = topo_2dlist[cnt]
                    results = results2d[cnt]
                    if len(other_topo_2ddlist) != 0:
                        other_topo_dlist = other_topo_2ddlist[cnt]
                    else: 
                        other_topo_dlist = []
                    model3d_dict[utime] = [topo_ls, results, other_topo_dlist]
                    
            self.model3d_dict = model3d_dict
        
        def search(self):
            #get the date the user is searching for
            search_date = self.params2.param('Parm3d').param("Search Date").value()
            search_datetime = parse(search_date)
            search_unix_time = time.mktime(search_datetime.timetuple())
            
            topo_unixtime_sorted = self.topo_unixtime_sorted
            diff_ls = []
            for cnt,ut in enumerate(topo_unixtime_sorted):
                diff = search_unix_time - ut
                diff = abs(diff)
                diff_ls.append(diff)
            
            mn = min(diff_ls)
            new_index = diff_ls.index(mn)
            self.current_time_index = new_index
            self.load_3dmodel()
            
            res_str = 'The result is ' + str(round(mn/60,2)) + 'mins from the search'
            self.params2.param('Parm3d').param("Search Result").setValue(res_str)
            
        def previous_scene(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            if current_time_index == 0:
                new_index = n3dtimes-1
            else:
                new_index = current_time_index - 1
            
            self.current_time_index = new_index

            self.load_3dmodel()
        
        def next_scene(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            if current_time_index == n3dtimes-1:
                new_index = 0
            else:
                new_index = current_time_index + 1
            
            self.current_time_index = new_index
            self.load_3dmodel()
            
        def update(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            rewind_status = self.rewind_status
            
            if rewind_status == True:
                if current_time_index == 0:
                    new_index = n3dtimes-1
                else:
                    new_index = current_time_index - 1
                    
            else:
                if current_time_index == n3dtimes-1:
                    new_index = 0
                else:
                    new_index = current_time_index + 1
            
            self.current_time_index = new_index
            self.load_3dmodel()
            
        def pause(self):
            cur_status = self.timer_status
            status_str = ""
            if cur_status:
                self.timer.stop()
                self.timer_status = False
                status_str += "Pause"
            else:
                self.timer.start(self.playback_speed)
                self.timer_status = True
                status_str += "Play"
            
            rewind_state = self.rewind_status
            if rewind_state :
                status_str += "(Rewind)"
                self.params2.param('Animate Parms').param('Play Status').setValue(status_str)
            else:
                status_str += "(Forward)"
                self.params2.param('Animate Parms').param('Play Status').setValue(status_str)
        
        def rewind(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
            
            self.rewind_status = True
            self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
        
        def forward(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
            
            self.rewind_status = False
            self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
        
        def change_speed(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
                
            seconds = self.params2.param('Animate Parms').param('Seconds/Frame').value()
            self.playback_speed = int(seconds*1000)
            
            rewind_state = self.rewind_status
            if rewind_state :
                self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
            else:
                self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
        
        def clear_3dview(self):
            view_3d = self.view3d
            all_items = view_3d.items
            nitems = len(all_items)
            while (nitems !=0):
                for i in all_items:
                    view_3d.removeItem(i)
                
                all_items = view_3d.items
                nitems = len(all_items)
        
        def animate(self):
            timer = QtCore.QTimer()
            self.timer = timer
            timer.timeout.connect(self.update)
            # timer.start(self.playback_speed)
            self.timer_status = False
            self.rewind_status = False
            self.playback_speed = 1000
            self.start()
                
        def start(self):
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                pg.exec()
    #--------------------------------------------------------------------------------------------
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    #convert the datetimes to unix timestamp
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = time.mktime(dt.timetuple())
        unix43d.append(unix_time)
    
    pg.mkQApp()
    win = AFalsecolour(topo_2dlist, results2d, unix43d, false_min_max_val = false_min_max_val, 
                       other_topo_2ddlist = other_topo_2ddlist)
    
    win.setWindowTitle("AnimateFalseColour")
    win.show()
    win.resize(1100,700)
    win.animate()

def viz_animate(topo_2ddlist, topo_datetime_ls):
    """
    This function produces an animation window.
 
    Parameters
    ----------
    topo_2ddlist : a 2d list of dictionary, optional
        A 2d list of dictionary specifying the visualisation parameters.
        topo_list: the topos to visualise
        colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        draw_edges: bool whether to draw the edges of mesh, default is True
        point_size: size of the point
        px_mode: True or False, if true the size of the point is in pixel if not its in meters
    
    topo_datetime_ls : list of datetime
        timestamp of the topology data.
    
    """
    class AnimateTopo(QtWidgets.QWidget):
        def __init__(self, topo_2ddlist, topo_unixtime):
            QtWidgets.QWidget.__init__(self)
            self.setupGUI()
            self.topo_2ddlist = topo_2ddlist
            self.topo_unixtime = topo_unixtime

            self.insert_parm3d()
            # check if there are multiple sets of geometry or just one set of geometry with multiple results
            self.create_3d_dict()
            self.topo_unixtime_sorted = sorted(self.model3d_dict.keys())
            self.current_time_index = 0
            self.n3dtimes = len(self.topo_unixtime_sorted)
            
            start_3d_unixt = self.topo_unixtime_sorted[0]
            end_3d_unixt = self.topo_unixtime_sorted[-1]
            date_str_start = datetime.datetime.fromtimestamp(start_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
            date_str_end = datetime.datetime.fromtimestamp(end_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
            range_str = date_str_start + ' to ' + date_str_end
            self.params2.param('Parm3d').param('Data Range Loaded').setValue(range_str)
            #load the initial spatial time-series data
            self.load_3dmodel()
            
        def setupGUI(self):
            self.layout = QtWidgets.QVBoxLayout()
            self.setLayout(self.layout)
            #create the right panel
            self.splitter = QtWidgets.QSplitter()
            self.splitter.setOrientation(QtCore.Qt.Orientation.Horizontal)
            #create the parameter tree for housing the parameters
            self.tree = ParameterTree(showHeader=False)
            self.splitter.addWidget(self.tree)
            #create the 3d view
            self.view3d = gl.GLViewWidget()
            self.splitter.addWidget(self.view3d)
            self.layout.addWidget(self.splitter)
            
            self.screen_width = self.screen().size().width()
            self.screen_height = self.screen().size().height()
            self.splitter.setSizes([int(self.screen_width/3.5), self.screen_width])
            
            self.playback_speed = 1000
            
        def insert_parm3d(self):
            self.parm3d = dict(name='Parm3d', type='group', expanded = True, title = "3D View Controls",
                       children=[
                                    dict(name='Data Range Loaded', type = 'str', title = "Data Range Loaded", readonly = True),
                                    dict(name='Current Date', type = 'str', title = "Current Date", readonly = True),
                                    dict(name='Search Date', type = 'str', title = "Search Date", value = "yyyy-mm-ddTHH:MM:SS"),
                                    dict(name='Search', type = 'action', title = "Search"),
                                    dict(name='Search Result', type = 'str', value = '', readonly = True),
                                    dict(name='Previous', type = 'action', title = "Previous"),
                                    dict(name='Next', type = 'action', title = "Next")
                                ]
                        )
            self.animate_parm = dict(name='Animate Parms', type='group', expanded = False, title = "Animate Parameters",
                       children=[
                                   dict(name='Play Status', type = 'str', title = "Play Status", value = 'Pause(Forward)', readonly=True),
                                   dict(name='Pause/Play', type = 'action', title = "Pause/Play"),
                                   dict(name='Rewind', type = 'action', title = "Rewind"),
                                   dict(name='Forward', type = 'action', title = "Forward"),
                                   dict(name='Seconds/Frame', type = 'float', title = "Seconds/Frame", value = 1.0),
                                   dict(name='Change Playback Speed', type = 'action', title = "Change Playback Speed")
                                ]
                        )
            self.params2 = Parameter.create(name = "View3dparm", type = "group", children = [self.parm3d, self.animate_parm])
            self.tree.addParameters(self.params2, showTop=False)
            self.params2.param('Parm3d').param("Search").sigActivated.connect(self.search)
            self.params2.param('Parm3d').param("Previous").sigActivated.connect(self.previous_scene)
            self.params2.param('Parm3d').param("Next").sigActivated.connect(self.next_scene)
            self.params2.param('Animate Parms').param("Pause/Play").sigActivated.connect(self.pause)
            self.params2.param('Animate Parms').param("Rewind").sigActivated.connect(self.rewind)
            self.params2.param('Animate Parms').param("Forward").sigActivated.connect(self.forward)
            self.params2.param('Animate Parms').param("Change Playback Speed").sigActivated.connect(self.change_speed)
        
        def load_3dmodel(self):
            #clear the 3dview
            self.clear_3dview()
            model3d_dict = self.model3d_dict
            
            topo_unixtime_sorted = self.topo_unixtime_sorted
            current_time_index = self.current_time_index
            
            current_unix_time = topo_unixtime_sorted[current_time_index]
            topo_dlist  = model3d_dict[current_unix_time] 
            bbox_list = _convert_topo_dictionary_list4viz(topo_dlist, self.view3d)
            overall_bbox = calculate.bbox_frm_bboxes(bbox_list)
            midpt = calculate.bbox_centre(overall_bbox)
            self.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
            
            lwr_left = [overall_bbox.minx, overall_bbox.miny, overall_bbox.minz]
            upr_right =  [overall_bbox.maxx, overall_bbox.maxy, overall_bbox.maxz]
            dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
            self.view3d.opts['distance'] = dist*1.5
            
            # date_str = datetime.datetime.utcfromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
            date_str = datetime.datetime.fromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
            
            self.params2.param('Parm3d').param('Current Date').setValue(date_str)
            
        def create_3d_dict(self):
            topo_2ddlist = self.topo_2ddlist
            topo_unixtime = self.topo_unixtime
            
            model3d_dict = {}
        
            for cnt, utime in enumerate(topo_unixtime):
                topo_dlist = topo_2ddlist[cnt]
                model3d_dict[utime] = topo_dlist
                
            self.model3d_dict = model3d_dict
        
        def search(self):
            #get the date the user is searching for
            search_date = self.params2.param('Parm3d').param("Search Date").value()
            search_datetime = parse(search_date)
            search_unix_time = time.mktime(search_datetime.timetuple())
            
            topo_unixtime_sorted = self.topo_unixtime_sorted
            diff_ls = []
            for cnt,ut in enumerate(topo_unixtime_sorted):
                diff = search_unix_time - ut
                diff = abs(diff)
                diff_ls.append(diff)
            
            mn = min(diff_ls)
            new_index = diff_ls.index(mn)
            self.current_time_index = new_index
            self.load_3dmodel()
            
            res_str = 'The result is ' + str(round(mn/60,2)) + 'mins from the search'
            self.params2.param('Parm3d').param("Search Result").setValue(res_str)
            
        def previous_scene(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            if current_time_index == 0:
                new_index = n3dtimes-1
            else:
                new_index = current_time_index - 1
            
            self.current_time_index = new_index

            self.load_3dmodel()
        
        def next_scene(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            if current_time_index == n3dtimes-1:
                new_index = 0
            else:
                new_index = current_time_index + 1
            
            self.current_time_index = new_index
            self.load_3dmodel()
            
        def update(self):
            current_time_index = self.current_time_index
            n3dtimes = self.n3dtimes
            rewind_status = self.rewind_status
            
            if rewind_status == True:
                if current_time_index == 0:
                    new_index = n3dtimes-1
                else:
                    new_index = current_time_index - 1
                    
            else:
                if current_time_index == n3dtimes-1:
                    new_index = 0
                else:
                    new_index = current_time_index + 1
            
            self.current_time_index = new_index
            self.load_3dmodel()
            
        def pause(self):
            cur_status = self.timer_status
            status_str = ""
            if cur_status:
                self.timer.stop()
                self.timer_status = False
                status_str += "Pause"
            else:
                self.timer.start(self.playback_speed)
                self.timer_status = True
                status_str += "Play"
            
            rewind_state = self.rewind_status
            if rewind_state :
                status_str += "(Rewind)"
                self.params2.param('Animate Parms').param('Play Status').setValue(status_str)
            else:
                status_str += "(Forward)"
                self.params2.param('Animate Parms').param('Play Status').setValue(status_str)
        
        def rewind(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
            
            self.rewind_status = True
            self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
        
        def forward(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
            
            self.rewind_status = False
            self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
        
        def change_speed(self):
            cur_status = self.timer_status
            if cur_status:
                self.timer.stop()
                self.timer_status = False
                
            seconds = self.params2.param('Animate Parms').param('Seconds/Frame').value()
            self.playback_speed = int(seconds*1000)
            
            rewind_state = self.rewind_status
            if rewind_state :
                self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
            else:
                self.params2.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
        
        def clear_3dview(self):
            view_3d = self.view3d
            all_items = view_3d.items
            nitems = len(all_items)
            while (nitems !=0):
                for i in all_items:
                    view_3d.removeItem(i)
                
                all_items = view_3d.items
                nitems = len(all_items)
        
        def animate(self):
            timer = QtCore.QTimer()
            self.timer = timer
            timer.timeout.connect(self.update)
            # timer.start(self.playback_speed)
            self.timer_status = False
            self.rewind_status = False
            self.playback_speed = 1000
            self.start()
                
        def start(self):
            if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
                pg.exec()
    #========================================================================
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    
    #convert the datetimes to unix timestamp
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = time.mktime(dt.timetuple())
        unix43d.append(unix_time)
    
    pg.mkQApp()
    win = AnimateTopo(topo_2ddlist, unix43d)
    
    win.setWindowTitle("AnimateTopo")
    win.show()
    win.resize(1100,700)
    win.animate()

def viz_graph(xvalues2d, yvalues2d, colour_ls, xlabel = None, xunit = None, ylabel = None, yunit = None, title = None, 
              legend = None, inf_lines = None, regions = None):
    """
    This function visualises time-series data.
 
    Parameters
    ----------
    xvalues2d : 2dlist of floats/datetime
        xvalues to plot. You can input python datetime object for time-series data. For different set of data separate them into different list.
    
    yvalues2d : 2dlist of floats
        yvalues to plot. For different set of data separate them into different list.
    
    colour_ls : list of tuple
        list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)].
    
    xlabel : str, optional
        label for the x-axis.
    
    xunit : str, optional
        unit label for the x-axis.
        
    ylabel : str, optional
        label for the y-axis.
    
    yunit : str, optional
        unit label for the y-axis.
        
    title : str, optional
        title for the graph.
        
    legend : list of str, optional
        list of strings for the legends.
        
    inf_lines : list of dictionary, optional
        dictionary describing the infinite line.
        label: str to describe the infinite line
        angle: float, the angle of the infinite line, 0=horizontal, 90=vertical
        pos: float, for horizontal and vertical line a single value is sufficient, for slanted line (x,y) is required.
        colour: tuple, (r,g,b,a)
    
    regions : list of dictionary, optional
        dictionary describing the region on the graph.
        label: str to describe the region.
        orientation: str, vertical or horizontal
        range: list of floats, [lwr_limit, upr_limit] e.g. [50,70] 
        colour: tuple, (r,g,b,a)
        
    """
    class GraphView(QtWidgets.QWidget):
        def __init__(self, is_time_series):
            QtWidgets.QWidget.__init__(self)
            self.setupGUI(is_time_series)
            
        def setupGUI(self, is_time_series):
            self.layout = QtWidgets.QVBoxLayout()
            self.layout.setContentsMargins(0,0,0,0)
            self.setLayout(self.layout)
            
            self.plot = pg.plot()
            if is_time_series == True:
                self.plot.setAxisItems(axisItems = {'bottom': pg.DateAxisItem()})
            
            self.plot.showGrid(x=True, y=True)
            self.layout.addWidget(self.plot)
            
        def add_scatter(self, scatterplot_ls):
            self.scatterplot_ls = scatterplot_ls
            for scatter in scatterplot_ls:
                self.plot.addItem(scatter)
    #========================================================================
    pg.mkQApp()
    
    is_time_series = False
    if type(xvalues2d[0][0]) == datetime.datetime:
        #convert datetime to unix timestamp
        is_time_series = True
        unix_2d = _convert_datetime2unix(xvalues2d)
        xvalues2d = unix_2d
        
    win = GraphView(is_time_series)
    scatterplot_ls = _plot_graph(xvalues2d, yvalues2d, colour_ls)
    win.add_scatter(scatterplot_ls)
    win.plot.setTitle(title)
    win.plot.setLabel('bottom', text = xlabel, units = xunit)
    win.plot.setLabel('left', text = ylabel, units = yunit)
    
    legend_item = pg.LegendItem((80,60), offset=(70,20))
    legend_item.setParentItem(win.plot.graphicsItem())
    if legend is None:
        for cnt, scatter in enumerate(win.scatterplot_ls):
            legend_item.addItem(scatter, 'legend' + str(cnt))
    else:
        for cnt, scatter in enumerate(win.scatterplot_ls):
            legend_item.addItem(scatter, legend[cnt])
    
    if inf_lines is not None:
        for infd in inf_lines:
            label = infd['label']
            angle = infd['angle']
            init_pos = infd['pos']
            colour = infd['colour']
            if type(init_pos) == datetime.datetime:
                init_pos = time.mktime(init_pos.timetuple())
                
            infl = pg.InfiniteLine(movable=False, angle=angle, label=label, pos = init_pos, pen = colour,
                                   labelOpts={'position':0.9, 'color': (0,0,0), 'fill': (255,255,255,150), 
                                              'movable': True})
            win.plot.addItem(infl)
    
    if regions is not None:
        for reg in regions:
            label = reg['label']
            orient = reg['orientation']
            rangex = reg['range']
            colour = reg['colour']
            if type(rangex[0]) == datetime.datetime:
                rangex = [time.mktime(rangex[0].timetuple()),
                          time.mktime(rangex[1].timetuple())]
            
            lr = pg.LinearRegionItem(values=rangex, orientation=orient, brush=colour, pen=[255,255,255,80], movable=False)
            win.plot.addItem(lr)
            label = pg.InfLineLabel(lr.lines[1], label, position=0.9, rotateAxis=(1,0), anchor=(1, 1))

    win.setWindowTitle("GraphView")
    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()
            
def viz_vx_dict(vx_dict, colour, wireframe = True):
    """
    This function visualises the voxel dictionaries.
 
    Parameters
    ----------
    vx_dict : voxel dictionary generated form modify.xyzs2vox
        dictionary of voxels 
        vox_dim: the dimension of the voxels
        voxels: dictionary of voxels
    
    colour :  str or tuple 
        keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        
    wireframe : bool, optional
        default True. 
    
    """
    vox_dim = vx_dict['voxel_dim']
    voxs = vx_dict['voxels']
    viz_ls = []
    for cnt,key in enumerate(voxs.keys()):
        vx = voxs[key]
        midpt = vx['midpt']
        box = create.box(vox_dim[0], vox_dim[1], vox_dim[2], centre_pt=midpt)
        if wireframe == False:
            viz_ls.append(box)
        else:
            bedges = get.edges_frm_solid(box)
            viz_ls.extend(bedges)
            
    viz([{'topo_list':viz_ls, 'colour':colour}])
        
def _make_mesh(xyzs, indices, face_colours = [], shader = "shaded", 
              gloptions = "opaque", draw_edges = False, 
              edge_colours = [0,0,0,1]):
    """
    This function returns a Mesh Item that can be viz by pyqtgraph.
 
    Parameters
    ----------
    xyzs : ndarray of shape(Nvertices,3)
        ndarray of shape(Nvertices,3) of the mesh.
        
    indices : ndarray of shape(Ntriangles,3)
        indices of the mesh.
    
    face_colours : ndarray of shape(Ntrianges, 4), optional
        array of colours specifying the colours of each triangles of the mesh.
        
    shade : string, optional
        specify the shader visualisation of the mesh. The options are: 
        shaded, balloon, normalColor, viewNormalColor, edgeHilight, 
        heightColor
        
    gloptions : string, optional
        specify the gloption visualisation of the mesh. The options are: 
        additive, opaque, translucent
    
    draw_edges : bool, optional
        toggle to whether draw the edges of the mesh
        
    edge_colours: list, option
        list with four elements specifying the RGB and Transparency of the edges.
        
    Returns
    -------
    mesh : mesh object
        mesh for visualisation.
    """
    if face_colours == []:
        mesh = gl.GLMeshItem(vertexes = xyzs, faces = indices, 
                             faceColors = None,
                             edgeColor = edge_colours,
                             smooth = False,
                             drawEdges = draw_edges, 
                             shader = shader,
                             glOptions = gloptions)
    else:
        mesh = gl.GLMeshItem(vertexes = xyzs, faces = indices,
                             faceColors = face_colours,
                             edgeColor = edge_colours,
                             smooth = False,
                             drawEdges = draw_edges, 
                             shader = shader,
                             glOptions = gloptions)
    
    return mesh

def _make_line(xyzs, line_colour = (0,0,0,1), width = 1, antialias=True, mode="lines"):
    """
    This function returns a Line Item that can be viz by pyqtgraph.
 
    Parameters
    ----------
    xyzs : ndarray of shape(Nvertices,3)
        ndarray of shape(Nvertices,3) of the line.
        
    face_colour : ndarray of shape(4), optional
        array of colours specifying the colours of the line.
    
    width : float, optional
        the width of the line
        
    antialias: bool, option
        smooth line drawing
    
    mode: str, option
        ‘lines’: Each pair of vertexes draws a single line segment.
        ‘line_strip’: All vertexes are drawn as a continuous set of line segments
        
    Returns
    -------
    line : line object
        line for visualisation.
    """
    line = gl.GLLinePlotItem(pos=xyzs, color= line_colour, width=width, antialias=antialias, mode = mode)
    return line

def _make_points(xyzs, point_colours, sizes, pxMode = True):
    """
    This function returns points Item that can be viz by pyqtgraph.
 
    Parameters
    ----------
    xyzs : ndarray of shape(Nvertices,3)
        ndarray of shape(Nvertices,3) of the line.
        
    point_colour : ndarray of shape(4), optional
        array of colours specifying the colours of the line, if a single tuple is given all points assumes that colour.
    
    sizes : float, optional
        list of floats of points sizes, if single val is given all point assumes that size
        
    pxMode: bool, option
        if True size is measured in pixels, if False size is based on units of the xyzs
        
    Returns
    -------
    points : point object
        point for visualisation.
    """
    points = gl.GLScatterPlotItem(pos=xyzs, color=point_colours, size=sizes, pxMode = pxMode)
    return points

def _convert_topo_dictionary_list4viz(topo_dictionary_list, view3d):
    """
    This function visualises the topologies.
 
    Parameters
    ----------
    topo_dictionary_list : a list of dictionary
        A list of dictionary specifying the visualisation parameters.
        topo_list: the topos to visualise
        colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        draw_edges: bool whether to draw the edges of mesh, default is True
        point_size: size of the point
        px_mode: True or False, if true the size of the point is in pixel if not its in meters
        att: name of the att to visualise with the topologies
        
    view3d : pyqtgraph 3d view widget
        3d view to visualise the geometries
    """
    def colour2rgb(colour):
        if colour == 'red':
            rgb = (1,0,0,1)
        elif colour == 'orange':
            rgb = (1,0.65,0,1)
        elif colour == 'yellow':
            rgb = (1,1,0,1)
        elif colour == 'green':
            rgb = (0,1,0,1)
        elif colour == 'blue':
            rgb = (0,0,1,1)
        elif colour == 'black':
            rgb = (0,0,0,1)
        elif colour == 'white':
            rgb = (1,1,1,1)
        else:
            rgb = colour
        return rgb
    
    bbox_list = []
    for d in topo_dictionary_list:
        colour = d['colour']
        rgb = colour2rgb(colour)
        draw_edges = True
        if 'draw_edges' in d.keys():
            draw_edges = d['draw_edges']
        
        pt_size = 10
        if 'point_size' in d.keys():
            pt_size = d['point_size']
            
        px_mode = True
        if 'px_mode' in d.keys():
            px_mode = d['px_mode']
        
        topo_list = d['topo_list']
        cmp = create.composite(topo_list)
        sorted_d = get.unpack_composite(cmp)
        #=================================================================================
        #get all the topology that can viz as points
        #=================================================================================
        vertices = sorted_d['vertex']
        if len(vertices) > 0:
            points_vertices = np.array([v.point.xyz for v in vertices])
            viz_pts = _make_points(points_vertices, rgb, pt_size, pxMode = px_mode)
            view3d.addItem(viz_pts)
            
        #=================================================================================
        #get all the topology that can viz as lines
        #=================================================================================
        all_edges = []
        edges = sorted_d['edge']
        if len(edges) > 0:
            all_edges = edges
        
        wires = sorted_d['wire']
        if len(wires) > 0:
            wires2edges = np.array([get.edges_frm_wire(wire) for wire in wires], dtype=object)
            wires2edges = wires2edges.flatten()
            all_edges = np.append(all_edges, wires2edges )
        
        if len(all_edges) > 0:
            line_vertices = modify.edges2lines(all_edges)
            viz_lines = _make_line(line_vertices, line_colour = rgb)
            view3d.addItem(viz_lines)  
            
        #=================================================================================
        #get all the topology that can be viz as mesh
        #=================================================================================
        all_faces = []
        faces = sorted_d['face']
        if len(faces) > 0:
            all_faces = faces
        
        shells = sorted_d['shell']
        if len(shells) > 0:
            shells2faces = np.array([get.faces_frm_shell(shell) for shell in shells], dtype=object)
            shells2faces = list(chain(*shells2faces)) #shells2faces.flatten()
            all_faces = np.append(all_faces, shells2faces)
        
        solids = sorted_d['solid']
        if len(solids) > 0:
            solids2faces = np.array([get.faces_frm_solid(solid) for solid in solids], dtype=object)
            solids2faces = list(chain(*solids2faces))#solids2faces.flatten()
            all_faces = np.append(all_faces, solids2faces)
        
        #if there are faces to be viz
        if len(all_faces) > 0:
            mesh_dict = modify.faces2mesh(all_faces)
            #flip the indices
            verts = mesh_dict['vertices']
            idx = mesh_dict['indices']
            #flip the vertices to be clockwise
            idx = np.flip(idx, axis=1)
            viz_mesh = _make_mesh(verts, idx,  draw_edges = draw_edges)
            viz_mesh.setColor(np.array(rgb))
            view3d.addItem(viz_mesh)     
        #=================================================================================
        #find the bbox
        #=================================================================================
        bbox = calculate.bbox_frm_topo(cmp)
        bbox_list.append(bbox)
    
    return bbox_list

def _sort_topos(topo, topo_type, icnt, topo_int_ls):
    """
    This function sorts the topology into topo_int_ls for falsecolour visualization.
 
    Parameters
    ----------
    topo : topobj
        the topo to sort.
        
    topo_type : topobj.TopoType
        array of colours specifying the colours of the line, if a single tuple is given all points assumes that colour.
    
    icnt : int
        the interval the topo belongs to. The falsecolour viz is divided into 10 intervals. 0 being the lowest and 9 being the highest.
        
    topo_int_ls : multidimensional list
        the topo is sorted into the list. [[[pts ...],[edges ...],[faces ...]],[[[pts ...],[edges ...],[faces ...]]]]
    """
    if topo_type == topobj.TopoType.VERTEX:
        pnt = topo.point.xyz
        topo_int_ls[icnt][0].append(pnt)
    elif topo_type == topobj.TopoType.EDGE:
        topo_int_ls[icnt][1].append(topo)
    elif topo_type == topobj.TopoType.WIRE:
        edges = get.edges_frm_wire(topo)
        topo_int_ls[icnt][1].extend(edges)
    elif topo_type == topobj.TopoType.FACE:
        topo_int_ls[icnt][2].append(topo)
    elif topo_type == topobj.TopoType.SHELL:
        faces = get.faces_frm_shell(topo)
        topo_int_ls[icnt][2].extend(faces)
    elif topo_type == topobj.TopoType.SOLID:
        faces = get.faces_frm_solid(topo)
        topo_int_ls[icnt][2].extend(faces)
    elif topo_type == topobj.TopoType.COMPOSITE:
        sorted_d = get.unpack_composite(topo)
        vertices = sorted_d['vertex']
        if len(vertices) > 0:
            pnts = [v.point.xyz for v in vertices]
            topo_int_ls[icnt][0].extend(pnts)
        #=================================================================================
        #get all the topology that can viz as lines
        #=================================================================================
        cedges = sorted_d['edge']
        if len(cedges) > 0:
            topo_int_ls[icnt][1].extend(cedges)
        wires = sorted_d['wire']
        if len(wires) > 0:
            edges = np.array([get.edges_frm_wire(wire) for wire in wires])
            edges = edges.flatten()
            topo_int_ls[icnt][1].extend(edges.tolist())
        #=================================================================================
        #get all the topology that can be viz as mesh
        #=================================================================================
        cfaces = sorted_d['face']
        if len(cfaces) > 0:
            topo_int_ls[icnt][2].extend(cfaces)
        
        shells = sorted_d['shell']
        if len(shells) > 0:
            shells2faces = np.array([get.faces_frm_shell(shell) for shell in shells])
            shells2faces = shells2faces.flatten()
            topo_int_ls[icnt][2].extend(shells2faces.tolist())
        
        solids = sorted_d['solid']
        if len(solids) > 0:
            solids2faces = np.array([get.faces_frm_solid(solid) for solid in solids])
            solids2faces = solids2faces.flatten()
            topo_int_ls[icnt][2].extend(solids2faces.tolist())

def _create_topo_int_ls(int_ls):
    """
    This function creates a topo ls for the falsecolour and spatial time-series .
 
    Parameters
    ----------
    int_ls : 2d list of tuple
        2d list of tuple specifying the [min, max] of each interval.
        
    Returns
    -------
    topo_int_ls : multidimensional list
        multidimensional empty list ready to be populated with topobj. [[[pts ...],[edges ...],[faces ...]], 
                                                                        [[[pts ...],[edges ...],[faces ...]]]]
    """
    topo_int_ls = []
    for _ in range(len(int_ls)):
        topo_int_ls.append([])
        topo_int_ls[-1].append([])
        topo_int_ls[-1].append([])
        topo_int_ls[-1].append([])
    return topo_int_ls

def _convert_datetime2unix(datetime2d):
    """
    This function converts a 2d list of datetime object into unix timestamps.
 
    Parameters
    ----------
    datetime2d : 2d list of datetime object
        datetimes to be converted.
        
    Returns
    -------
    unix2d : 2d list of unix timestamps
        the converted unix timestamps.
    """
    unix2d = []
    for xvalues in datetime2d:
        unix_ls = []
        for x in xvalues:
            unix_time = time.mktime(x.timetuple())
            unix_ls.append(unix_time)
        unix2d.append(unix_ls)
    return unix2d

def _plot_graph(xvalues2d, yvalues2d, colour_ls):
    scatterplot_ls = []
    for cnt, xvalues in enumerate(xvalues2d):
        yvalues = yvalues2d[cnt]
        all_vals = [xvalues, yvalues]
        zip_vals = list(zip(*all_vals))
        spots = []
        for zval in zip_vals:
            spots.append({'pos': zval, 'size':10})
        
        scatterplot = pg.ScatterPlotItem(size = 10, hoverable = True, 
                                         brush= pg.mkBrush(colour_ls[cnt][0], colour_ls[cnt][1],colour_ls[cnt][2], colour_ls[cnt][3]), 
                                         hoverSymbol='s', hoverPen=pg.mkPen('r', width=2))
        scatterplot.addPoints(spots)
        scatterplot_ls.append(scatterplot)
    return scatterplot_ls
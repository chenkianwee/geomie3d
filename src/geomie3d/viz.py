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

# setup the QT environment
# import PyQT6
os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt6'
if sys.platform == 'linux' or sys.platform == 'linux2':
    # os.environ['QT_QPA_PLATFORM'] = 'wayland'
    # os.environ['QT_QPA_PLATFORM'] = 'wayland-egl'
    os.environ['PYOPENGL_PLATFORM'] = 'egl'

QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

class BaseAnimate(QtWidgets.QWidget):
    def __init__(self, topo_unixtime: list[float], gl_option: str):
        """
        the base class for all animation visualization
        """
        QtWidgets.QWidget.__init__(self)
        self.topo_unixtime: list[float] = topo_unixtime
        """the timestamp of the datetime"""
        self.gl_option: str = gl_option
        """str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent'"""
        self.current_time_index: int = 0
        """index of the current time"""
        
    def setupGUI(self):
        """
        function for setting up the base animation gui
        """
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
           """
           function for setting up the base parameters of the gui
           """
           self.parm3d = dict(name='Parm3d', type='group', expanded = True, title = "3D View Controls",
                      children=[
                                   dict(name='Data Range Loaded', type = 'str', title = "Data Range Loaded", default = '', readonly = True),
                                   dict(name='Current Date', type = 'str', title = "Current Date", default = '', readonly = True),
                                   dict(name='Search Date', type = 'str', title = "Search Date", default = "yyyy-mm-ddTHH:MM:SS"),
                                   dict(name='Search', type = 'action', title = "Search"),
                                   dict(name='Search Result', type = 'str', default = '', readonly = True),
                                   dict(name='Next', type = 'action', title = "Next"),
                                   dict(name='Previous', type = 'action', title = "Previous")
                               ]
                       )
           self.animate_parm = dict(name='Animate Parms', type='group', expanded = False, title = "Animate Parameters",
                      children=[
                                  dict(name='Play Status', type = 'str', title = "Play Status", default = 'Pause(Forward)', readonly=True),
                                  dict(name='Pause/Play', type = 'action', title = "Pause/Play"),
                                  dict(name='Forward', type = 'action', title = "Forward"),
                                  dict(name='Rewind', type = 'action', title = "Rewind"),
                                  dict(name='Seconds/Frame', type = 'float', title = "Seconds/Frame", default = 1.0),
                                  dict(name='Change Playback Speed', type = 'action', title = "Change Playback Speed")
                               ]
                       )
           self.params = Parameter.create(name = "View3dparm", type = "group", children = [self.parm3d, self.animate_parm])
           self.tree.addParameters(self.params, showTop=False)
           self.params.param('Parm3d').param("Search").sigActivated.connect(self.search)
           self.params.param('Parm3d').param("Previous").sigActivated.connect(self.previous_scene)
           self.params.param('Parm3d').param("Next").sigActivated.connect(self.next_scene)
           self.params.param('Animate Parms').param("Pause/Play").sigActivated.connect(self.pause)
           self.params.param('Animate Parms').param("Rewind").sigActivated.connect(self.rewind)
           self.params.param('Animate Parms').param("Forward").sigActivated.connect(self.forward)
           self.params.param('Animate Parms').param("Change Playback Speed").sigActivated.connect(self.change_speed)
    
    def gen_falsecolour_bar(self, user_min: float, user_max: float, results: list[float]) -> tuple[np.ndarray, gl.GLGradientLegendItem]:
        """
        This function sorts the normalized results into 10 equal intervals.
    
        Parameters
        ----------
        user_min : float
            the minimum of the falsecolor bar.

        user_max : float
            the maximum of the falsecolor bar.

        results : list[float]
            the result values.
        
        Returns
        -------
        colors : np.ndarray
            array of floats between 0-1.
        
        gll : gl.GLGradientLegendItem
            gradient object for inserting into the 3d view port.

        """
        real_min = self.real_min_max[0]
        real_max = self.real_min_max[1]
        colours, gll, bcolour = _map_gen_bar(user_min, user_max, real_min, real_max, results)
        self.colour_refs = bcolour
        return colours, gll
    
    def search(self):
        """
        provides search function to the gui
        """
        #get the date the user is searching for
        search_date = self.params.param('Parm3d').param("Search Date").value()
        search_datetime = parse(search_date)
        search_unix_time = search_datetime.timestamp()
        
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
        self.params.param('Parm3d').param("Search Result").setValue(res_str)
        
    def previous_scene(self):
        """
        move to the previous scene
        """
        current_time_index = self.current_time_index
        n3dtimes = self.n3dtimes
        if current_time_index == 0:
            new_index = n3dtimes-1
        else:
            new_index = current_time_index - 1
        
        self.current_time_index = new_index

        self.load_3dmodel()
    
    def next_scene(self):
        """
        move to the next scene
        """
        current_time_index = self.current_time_index
        n3dtimes = self.n3dtimes
        if current_time_index == n3dtimes-1:
            new_index = 0
        else:
            new_index = current_time_index + 1
        
        self.current_time_index = new_index
        self.load_3dmodel()
        
    def update(self):
        """
        updates the 3d view port.
        """
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
        """
        pause the playing animation.
        """
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
            self.params.param('Animate Parms').param('Play Status').setValue(status_str)
        else:
            status_str += "(Forward)"
            self.params.param('Animate Parms').param('Play Status').setValue(status_str)
    
    def rewind(self):
        """
        rewinds the animation.
        """
        cur_status = self.timer_status
        if cur_status:
            self.timer.stop()
            self.timer_status = False
        
        self.rewind_status = True
        self.params.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
    
    def forward(self):
        """
        forwards the animation.
        """
        cur_status = self.timer_status
        if cur_status:
            self.timer.stop()
            self.timer_status = False
        
        self.rewind_status = False
        self.params.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
    
    def change_speed(self):
        """
        change the playback speed of the animation
        """
        cur_status = self.timer_status
        if cur_status:
            self.timer.stop()
            self.timer_status = False
            
        seconds = self.params.param('Animate Parms').param('Seconds/Frame').value()
        self.playback_speed = int(seconds*1000)
        
        rewind_state = self.rewind_status
        if rewind_state :
            self.params.param('Animate Parms').param('Play Status').setValue("Pause(Rewind)")
        else:
            self.params.param('Animate Parms').param('Play Status').setValue("Pause(Forward)")
    
    def clear_3dview(self):
        """
        clears the 3d view port.
        """
        view_3d = self.view3d
        all_items = view_3d.items
        nitems = len(all_items)
        while (nitems !=0):
            for i in all_items:
                view_3d.removeItem(i)
            
            all_items = view_3d.items
            nitems = len(all_items)
    
    def animate(self):
        """
        animate the scenes
        """
        timer = QtCore.QTimer()
        self.timer = timer
        timer.timeout.connect(self.update)
        # timer.start(self.playback_speed)
        self.timer_status = False
        self.rewind_status = False
        self.playback_speed = 1000
        self.start()
            
    def start(self):
        """
        start the gui
        """
        if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
            pg.exec()

class AFalsecolour(BaseAnimate):
    def __init__(self, topo_2dlist: list[list[topobj.Topology]], results2d: list[list[float]], topo_unixtime: list[float], 
                 false_min_max_val: list[float] = None, other_topo_2ddlist: list[list[dict]] = [], gl_option: str = 'opaque'):
        """
        Initializes a falsecolour animation window.
    
        Parameters
        ----------
        topo_2dlist : list[list[topobj.Topology]]
            2dlist of topologies. You can have one set of geometries to multiple sets of results.
        
        results2d : list[list[float]]
            2d list of performance value to be visualise as falsecolour.
        
        topo_datetime_ls : list[datetime.datetime]
            timestamp of the topology data.
        
        false_min_max_val : list[float], optional
            list of two values [min_val, max_val] for the falsecolour visualisation.
        
        other_topo_2ddlist : list[list[dict]], optional
            A 2d list of dictionary specifying the visualisation parameters.
            - topo_list: the topos to visualise
            - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
            - draw_edges: bool whether to draw the edges of mesh, default is True
            - point_size: size of the point
            - px_mode: True or False, if true the size of the point is in pixel if not its in meters
            - attribute: name of the attribute to viz
        
        gl_option : str, optional
            str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
        """
        super(AFalsecolour, self).__init__(topo_unixtime, gl_option)
        self.setupGUI()
        self.insert_parm3d()
        self.topo_2dlist: list[list[topobj.Topology]] = topo_2dlist
        """2dlist of topologies. You can have one set of geometries to multiple sets of results."""
        self.results2d: list[list[float]] = results2d
        """2d list of performance value to be visualise as falsecolour."""
        self.false_min_max_val: list[float] = false_min_max_val
        """list of two values [min_val, max_val] for the falsecolour visualisation."""
        self.other_topo_2ddlist: list[list[dict]] = other_topo_2ddlist
        """A 2d list of dictionary specifying the visualisation parameters."""
        #process data to visualise 3d model
        res_flat = list(chain(*results2d))
        self.real_min_max: list[float] = [min(res_flat), max(res_flat)]
        """the min max value of the results"""
        self.create_3d_dict()
        self.topo_unixtime_sorted: list[float] = sorted(self.model3d_dict.keys())
        """timestamp of the topology data arranged in chronological order."""
        self.current_time_index: int = 0
        """index of the current timestamp"""
        self.n3dtimes: int = len(self.topo_unixtime_sorted)
        """the number of timestamps"""
        
        start_3d_unixt = self.topo_unixtime_sorted[0]
        end_3d_unixt = self.topo_unixtime_sorted[-1]
        date_str_start = datetime.datetime.fromtimestamp(start_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
        date_str_end = datetime.datetime.fromtimestamp(end_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
        range_str = date_str_start + ' to ' + date_str_end
        self.params.param('Parm3d').param('Data Range Loaded').setValue(range_str)
        #load the initial spatial time-series data
        self.load_3dmodel(zoom_extent = True)
    
    def load_3dmodel(self, zoom_extent: bool = False):
        """
        loads the 3d model onto the 3d view port.

        Parameters
        ----------
        zoom_extent : bool, optional
            zoom to the 3d model. Default = False.
        """
        #clear the 3dview
        self.clear_3dview()
        model3d_dict = self.model3d_dict
        
        topo_unixtime_sorted = self.topo_unixtime_sorted
        current_time_index = self.current_time_index
        
        current_unix_time = topo_unixtime_sorted[current_time_index]
        model_res = model3d_dict[current_unix_time] 
        topo_list = model_res[0]
        results = model_res[1]
        other_topo_dlist = model_res[2]
        
        false_min_max_val = self.false_min_max_val
        if false_min_max_val == None:
            colours, legendbar = self.gen_falsecolour_bar(self.real_min_max[0], self.real_min_max[0], results)
        else:
            colours, legendbar = self.gen_falsecolour_bar(false_min_max_val[0], false_min_max_val[1], results)
            
        colour_refs = self.colour_refs
        
        clr_ints = _sort_topos2clr_ints(topo_list, colours, colour_refs)
        
        for cnt,clr_int in enumerate(clr_ints):
            colour = colour_refs[cnt*2]
            _clr_topos(clr_int, colour, self.view3d, gl_option = self.gl_option)
        
        self.view3d.addItem(gl.GLAxisItem())
        if zoom_extent == True:
            cmp = create.composite(topo_list)
            bbox = calculate.bbox_frm_topo(cmp)
            midpt = calculate.bboxes_centre([bbox])[0]
            self.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
            
            
            lwr_left = [bbox.minx, bbox.miny, bbox.minz]
            upr_right =  [bbox.maxx, bbox.maxy, bbox.maxz]
            dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
            self.view3d.opts['distance'] = dist*1.5
        
        if len(other_topo_dlist) != 0:
            _convert_topo_dictionary_list4viz(other_topo_dlist, self.view3d, gl_option = self.gl_option)
            
        self.view3d.addItem(legendbar)
        date_str = datetime.datetime.fromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
        
        self.params.param('Parm3d').param('Current Date').setValue(date_str)
        
    def create_3d_dict(self):
        """
        create a dictionary of all the topologies arrange according to their timestamp
        """
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

class AnimateTopo(BaseAnimate):
    def __init__(self, topo_2ddlist: list[list[dict]], topo_unixtime: list[float], gl_option: str = 'opaque'):
        """
        Produces an animation window.
    
        Parameters
        ----------
        topo_2ddlist : list[list[dict]]
            A 2d list of dictionary specifying the visualisation parameters.
            - topo_list: the topos to visualise
            - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
            - draw_edges: bool whether to draw the edges of mesh, default is True
            - point_size: size of the point
            - px_mode: True or False, if true the size of the point is in pixel if not its in meters
            - attribute: name of the attribute to viz
        
        topo_datetime_ls : list[datetime.datetime]
            timestamp of the topology data.
        
        gl_option : str, optional
            str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
        """
        super(AnimateTopo, self).__init__(topo_unixtime, gl_option)
        self.setupGUI()
        self.insert_parm3d()
        # check if there are multiple sets of geometry or just one set of geometry with multiple results
        self.topo_2ddlist: list[list[dict]] = topo_2ddlist
        """2dlist of topologies."""
        self.create_3d_dict()
        self.topo_unixtime_sorted: list[float] = sorted(self.model3d_dict.keys())
        """timestamp of the topology data arranged in chronological order."""
        self.n3dtimes: int = len(self.topo_unixtime_sorted)
        """the number of timestamps"""

        start_3d_unixt = self.topo_unixtime_sorted[0]
        end_3d_unixt = self.topo_unixtime_sorted[-1]
        date_str_start = datetime.datetime.fromtimestamp(start_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
        date_str_end = datetime.datetime.fromtimestamp(end_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
        range_str = date_str_start + ' to ' + date_str_end
        self.params.param('Parm3d').param('Data Range Loaded').setValue(range_str)
        self.load_3dmodel(zoom_extent=True)
        
    def load_3dmodel(self, zoom_extent: bool = False):
        """
        loads the 3d model onto the 3d view port.

        Parameters
        ----------
        zoom_extent : bool, optional
            zoom to the 3d model. Default = False.
        """
        #clear the 3dview
        self.clear_3dview()
        model3d_dict = self.model3d_dict
        
        topo_unixtime_sorted = self.topo_unixtime_sorted
        current_time_index = self.current_time_index
        
        current_unix_time = topo_unixtime_sorted[current_time_index]
        topo_dlist  = model3d_dict[current_unix_time] 
        bbox_list = _convert_topo_dictionary_list4viz(topo_dlist, self.view3d, gl_option = self.gl_option)
        if zoom_extent == True:
            overall_bbox = calculate.bbox_frm_bboxes(bbox_list)
            midpt = calculate.bboxes_centre([overall_bbox])[0]
            self.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
            
            lwr_left = [overall_bbox.minx, overall_bbox.miny, overall_bbox.minz]
            upr_right =  [overall_bbox.maxx, overall_bbox.maxy, overall_bbox.maxz]
            dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
            self.view3d.opts['distance'] = dist*1.5
        
        date_str = datetime.datetime.fromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
        
        self.params.param('Parm3d').param('Current Date').setValue(date_str)
        
    def create_3d_dict(self):
        """
        create a dictionary of all the topologies arrange according to their timestamp
        """
        topo_2ddlist = self.topo_2ddlist
        topo_unixtime = self.topo_unixtime
        
        model3d_dict = {}
    
        for cnt, utime in enumerate(topo_unixtime):
            topo_dlist = topo_2ddlist[cnt]
            model3d_dict[utime] = topo_dlist
            
        self.model3d_dict = model3d_dict

class FalsecolourView(QtWidgets.QWidget):
    def __init__(self):
        """
        gui object for visualizing topologies with falsecolor bar
        """
        QtWidgets.QWidget.__init__(self)
        self.real_min_max: list[float] = None
        """list specifying the min max [min, max]"""
        self.setupGUI()
        
    def setupGUI(self):
        """
        function for setting up the base gui
        """
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
        
        self.screen_width = self.screen().size().width()
        self.screen_height = self.screen().size().height()
        self.splitter.setSizes([int(self.screen_width/6), self.screen_width])
    
    def insert_export_parm(self):
        self.export3d = dict(name='Export3d', type='group', expanded = False, title = "Export Image",
                                children=[ dict(name = 'Export', type = 'action')])
        
        self.params = Parameter.create(name = "Export", type = "group", children = [self.export3d])
        self.params.param('Export3d').param('Export').sigActivated.connect(self.export)
        self.tree.addParameters(self.params, showTop=False)
    
    def export(self):
        """
        function for setting up the export parameters of the gui
        """
        fn = pg.FileDialog.getSaveFileName(self, "Choose File Path", "exported_img.png", 'PNG (*.png);; JPEG (*.jpg)')
        
        if fn == '':
            return

        res_path = str(fn[0])
        self.view3d.readQImage().save(res_path)
        
    def gen_falsecolour_bar(self, user_min: float, user_max: float, results: list[float]) -> tuple[np.ndarray, gl.GLGradientLegendItem]:
        """
        Generate the falsecolor bar.
    
        Parameters
        ----------
        user_min : float
            the minimum of the falsecolor bar.

        user_max : float
            the maximum of the falsecolor bar.

        results : list[float]
            the result values.
        
        Returns
        -------
        colors : np.ndarray
            array of floats between 0-1.
        
        gll : gl.GLGradientLegendItem
            gradient object for inserting into the 3d view port.

        """
        real_min = self.real_min_max[0]
        real_max = self.real_min_max[1]
        colours, gll, bcolour = _map_gen_bar(user_min, user_max, real_min, real_max, results)
        self.colour_refs = bcolour
        return colours, gll

class GraphView(QtWidgets.QWidget):
    def __init__(self, is_time_series: bool):
        """
        loads a graph view gui.

        Parameters
        ----------
        is_time_series : bool
            whether the data set is time-series.
        """
        QtWidgets.QWidget.__init__(self)
        self.setupGUI(is_time_series)
        
    def setupGUI(self, is_time_series: bool):
        """
        setup the gui of the graph

        Parameters
        ----------
        is_time_series : bool
            whether the data set is time-series.
        """
        self.layout = QtWidgets.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        
        self.plot = pg.plot()
        self.p1 = self.plot.plotItem
        if is_time_series == True:
            # self.p1.setAxisItems(axisItems = {'bottom': pg.DateAxisItem(utcOffset=time.timezone)})
            self.p1.setAxisItems(axisItems = {'bottom': pg.DateAxisItem()})
        
        self.plot.showGrid(x=True, y=True)
        self.layout.addWidget(self.plot)
        
    def add_scatter(self, scatterplot_ls: list[pg.ScatterPlotItem]):
        """
        add scatter points to the graph

        Parameters
        ----------
        scatterplot_ls : list[pg.ScatterPlotItem]
            scatterplot object to be plotted on the gui.
        """
        self.scatterplot_ls = scatterplot_ls
        for scatter in scatterplot_ls:
            self.p1.addItem(scatter)

class StView(BaseAnimate):
    def __init__(self, topo_2dlist: list[list[topobj.Topology]], results2d: list[list[float]], topo_unixtime: list[float], xvalues2d: list[list[float]], 
                    yvalues2d: list[list[float]], colour_ls: list[list[int]], false_min_max_val: list[float] = None, 
                    other_topo_2ddlist: list[list[dict]] = [], xlabel: str = None, xunit: str = None, ylabel: str = None, yunit: str = None, 
                    title: str = None, legend: list[str] = None, inf_lines: list[dict] = None, regions: list[dict] = None, 
                    second_xvalues2d: list[list[float]] = None, second_yvalues2d: list[list[float]] = None, second_colour_ls: list[list[int]] = None, 
                    second_legend: list[str] = None, second_ylabel: str = None, second_yunit: str = None, gl_option: str = 'opaque'):
        
        """
        Initialize the spatial time series gui object.
    
        Parameters
        ----------
        topo_2dlist : list[list[topobj.Topology]]
            2dlist of topologies. You can have one set of geometries to multiple sets of results.
        
        results2d : list[list[float]]
            2d list of performance value to be visualise as falsecolour.
        
        topo_unixtime : list[float]
            timestamp of the topology data.
        
        xvalues2d : list[list[datetime.datetime]]
            xvalues to plot. For different set of data separate them into different list.
        
        yvalues2d : list[list[float]]
            yvalues to plot. For different set of data separate them into different list.
        
        colour_ls : list[list[int]]
            list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)]. list[shape(number of colors, 3)] 
        
        false_min_max_val : list[float], optional
            list of two values [min_val, max_val] for the falsecolour visualisation.
        
        other_topo_2ddlist : list[list[dict]], optional
            A 2d list of dictionary specifying the visualisation parameters.
            - topo_list: the topos to visualise
            - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
            - draw_edges: bool whether to draw the edges of mesh, default is True
            - point_size: size of the point
            - px_mode: True or False, if true the size of the point is in pixel if not its in meters
            - attribute: name of the attribute to viz
        
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
        
        legend : list[str], optional
            list of strings for the legends.
            
        inf_lines : list[dict], optional
            dictionary describing the infinite line.
            - label: str to describe the infinite line
            - angle: float, the angle of the infinite line, 0=horizontal, 90=vertical
            - pos: float, for horizontal and vertical line a single value is sufficient, for slanted line (x,y) is required.
            - colour: tuple, (r,g,b,a)
            
        regions : list[dict], optional
            dictionary describing the region on the graph.
            - label: str to describe the region.
            - orientation: str, vertical or horizontal
            - range: list of floats, [lwr_limit, upr_limit] e.g. [50,70] 
            - colour: tuple, (r,g,b,a)
            
        second_xvalues2d : list[list[datetime.datetime]], optional
            xvalues to plot on a second y-axis. You can input python datetime object for time-series data. For different set of data separate them into different list.
            
        second_yvalues2d : list[list[float]], optional
            yvalues to plot on a second y-axis. For different set of data separate them into different list.
            
        second_colour_ls : list[list[int]], optional
            list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)].
        
        second_legend : list[str], optional
            list of strings for the data on the second axis.
        
        second_ylabel : str, optional
            label for the second y-axis.
        
        second_yunit : str, optional
            unit label for the second y-axis.
            
        gl_option : str, optional
            str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
        """
        super(StView, self).__init__(topo_unixtime, gl_option)
        self.setupGUI()
        self.insert_parm3d()

        self.topo_2dlist: list[list[topobj.Topology]] = topo_2dlist
        """2dlist of topologies. You can have one set of geometries to multiple sets of results."""
        self.results2d: list[list[float]] = results2d
        """2d list of performance value to be visualise as falsecolour."""
        self.xvalues2d: list[list[float]] = xvalues2d
        """xvalues to plot. For different set of data separate them into different list."""
        self.yvalues2d: list[list[float]] = yvalues2d
        """yvalues to plot. For different set of data separate them into different list."""
        self.colour_ls: list[list[int]] = colour_ls
        """list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)]. list[shape(number of colors, 3)]"""
        self.false_min_max_val: list[float] = false_min_max_val
        """list of two values [min_val, max_val] for the falsecolour visualisation."""
        self.other_topo_2ddlist: list[list[dict]] = other_topo_2ddlist
        """A 2d list of dictionary specifying the visualisation parameters."""
        self.xlabel: str = xlabel
        """label for the x-axis."""
        self.xunit: str = xunit
        """unit label for the x-axis."""
        self.ylabel: str = ylabel
        """label for the y-axis."""
        self.yunit: str = yunit
        """unit label for the y-axis."""
        self.title: str = title
        """title for the graph."""
        self.legend: list[str] = legend
        """list of strings for the legends."""
        self.inf_lines: list[dict] = inf_lines
        """dictionary describing the infinite line."""
        self.regions: list[dict] = regions
        """dictionary describing the region on the graph."""
        self.second_xvalues2d: list[list[float]] = second_xvalues2d
        """xvalues to plot on a second y-axis."""
        self.second_yvalues2d: list[list[float]] = second_yvalues2d
        """yvalues to plot on a second y-axis."""
        self.second_colour_ls: list[list[int]] = second_colour_ls
        """list of tuples specifying the colour."""
        self.second_legend: list[str] = second_legend
        """list of strings for the data on the second axis."""
        self.second_ylabel: str = second_ylabel
        """label for the second y-axis."""
        self.second_yunit: str = second_yunit
        """unit label for the second y-axis."""
        #process data to visualise 3d model
        res_flat = list(chain(*results2d))
        self.real_min_max: list[float] = [min(res_flat), max(res_flat)]
        """the min max value of the results"""
        # check if there are multiple sets of geometry or just one set of geometry with multiple results
        self.create_3d_dict()
        self.topo_unixtime_sorted: list[float] = sorted(self.model3d_dict.keys())
        """timestamp of the topology data arranged in chronological order."""
        self.current_time_index: int = 0
        """index of the current timestamp"""
        self.n3dtimes: int = len(self.topo_unixtime_sorted)
        """the number of timestamps"""
        
        start_3d_unixt = self.topo_unixtime_sorted[0]
        end_3d_unixt = self.topo_unixtime_sorted[-1]
        date_str_start = datetime.datetime.fromtimestamp(start_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
        date_str_end = datetime.datetime.fromtimestamp(end_3d_unixt).strftime('%Y-%m-%dT%H:%M:%S')
        range_str = date_str_start + ' to ' + date_str_end
        self.params.param('Parm3d').param('Data Range Loaded').setValue(range_str)
        #load the initial spatial time-series data
        self.load_3dmodel(zoom_extent=True)
        #load the time-series data
        self.load_time_series_data()
        
    def setupGUI(self):
        """
        function for setting up the base gui
        """
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
        self.p1 = self.plot.plotItem
        self.p1.setAxisItems(axisItems = {'bottom': pg.DateAxisItem()})
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
    
    def load_3dmodel(self, zoom_extent: bool = False):
        """
        loads the 3d model onto the 3d view port.

        Parameters
        ----------
        zoom_extent : bool, optional
            zoom to the 3d model. Default = False.
        """
        #clear the 3dview
        self.clear_3dview()
        model3d_dict = self.model3d_dict
        topo_unixtime_sorted = self.topo_unixtime_sorted
        current_time_index = self.current_time_index
        
        current_unix_time = topo_unixtime_sorted[current_time_index]
        model_res = model3d_dict[current_unix_time] 
        topo_list = model_res[0]
        results = model_res[1]
        other_topo_dlist = model_res[2]
        
        false_min_max_val = self.false_min_max_val
        if false_min_max_val == None:
            colours, legendbar = self.gen_falsecolour_bar(self.real_min_max[0], self.real_min_max[0], results)
        else:
            colours, legendbar = self.gen_falsecolour_bar(false_min_max_val[0], false_min_max_val[1], results)
            
        colour_refs = self.colour_refs
        
        clr_ints = _sort_topos2clr_ints(topo_list, colours, colour_refs, )
        
        for cnt,clr_int in enumerate(clr_ints):
            colour = colour_refs[cnt*2]
            _clr_topos(clr_int, colour, self.view3d, gl_option = self.gl_option)
            
            
        self.view3d.addItem(gl.GLAxisItem())
        if zoom_extent == True:
            cmp = create.composite(topo_list)
            bbox = calculate.bbox_frm_topo(cmp)
            midpt = calculate.bboxes_centre([bbox])[0]
            self.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
            
            lwr_left = [bbox.minx, bbox.miny, bbox.minz]
            upr_right =  [bbox.maxx, bbox.maxy, bbox.maxz]
            dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
            self.view3d.opts['distance'] = dist*1.5
        
        if len(other_topo_dlist) != 0:
            _convert_topo_dictionary_list4viz(other_topo_dlist, self.view3d, gl_option = self.gl_option)
        
        self.view3d.addItem(legendbar)
        date_str = datetime.datetime.fromtimestamp(current_unix_time).strftime('%Y-%m-%dT%H:%M:%S')
        self.params.param('Parm3d').param('Current Date').setValue(date_str)
        
    def create_3d_dict(self):
        """
        create a dictionary of all the topologies arrange according to their timestamp
        """
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
    
    def load_time_series_data(self):
        """
        load the time-series data onto the gui
        """
        p1 = self.p1
        self.scatterplot_ls = _plot_graph(self.xvalues2d, self.yvalues2d, self.colour_ls)
        legend_item = pg.LegendItem((80,60), offset=(70,20))
        legend_item.setParentItem(p1.graphicsItem())
        if self.legend is None:
            for cnt, scatter in enumerate(self.scatterplot_ls):
                p1.addItem(scatter)
                legend_item.addItem(scatter, 'legend' + str(cnt))
        else:
            for cnt, scatter in enumerate(self.scatterplot_ls):
                p1.addItem(scatter)
                legend_item.addItem(scatter, self.legend[cnt])
        
        #add an infinite line to the graph
        current_time_index = self.current_time_index
        current_unix_time = self.topo_unixtime_sorted[current_time_index]
        self.infl = pg.InfiniteLine(movable=False, angle=90, label='3DModelTime', pos = current_unix_time, pen = (255,255,255,255),
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
                    init_pos = init_pos.timestamp()
                    
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
                    rangex = [rangex[0].timestamp(), rangex[1].timestamp()]
                
                lr = pg.LinearRegionItem(values=rangex, orientation=orient, brush=colour, pen=[255,255,255,80], movable=False)
                label = pg.InfLineLabel(lr.lines[0], label, position=0.5, rotateAxis=(1,0), anchor=(1, 1))
                self.plot.addItem(lr)
        
        
        p1.addItem(self.infl)
        p1.setTitle(self.title)
        p1.setLabel('bottom', text = self.xlabel, units = self.xunit)
        p1.setLabel('left', text = self.ylabel, units = self.yunit)
        
        if self.second_xvalues2d is not None and self.second_yvalues2d is not None and self.second_colour_ls:
            #need to create a new second y-axis and link it
            ## create a new ViewBox, link the right axis to its coordinate system
            p2 = pg.ViewBox()
            p1.showAxis('right')
            p1.scene().addItem(p2)
            p1.getAxis('right').linkToView(p2)
            p2.setXLink(p1)
            p1.setLabel('right', text = self.second_ylabel, units = self.second_yunit)
            
            ## Handle view resizing 
            def updateViews():
                p2.setGeometry(p1.vb.sceneBoundingRect())
                p2.linkedViewChanged(p1.vb, p2.XAxis)
                
            updateViews()
            p1.vb.sigResized.connect(updateViews)
            
            scatter_ls2 = _plot_graph(self.second_xvalues2d, self.second_yvalues2d, self.second_colour_ls, symbol = 't')
            for scatter2 in scatter_ls2:
                p2.addItem(scatter2)
            
            if self.second_legend is None:
                for cnt, scatter2 in enumerate(scatter_ls2):
                    legend_item.addItem(scatter2, 'legend' + str(cnt))
            else:
                for cnt, scatter2 in enumerate(scatter_ls2):
                    legend_item.addItem(scatter2, self.second_legend[cnt])

    def update(self):
        """
        updates the all the 3d view ports.
        """
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
        current_time_index = self.current_time_index
        current_unix_time = self.topo_unixtime_sorted[current_time_index]
        self.infl.setValue(current_unix_time)
    
    def search(self):
        """
        provides search function to the gui
        """
        #get the date the user is searching for
        search_date = self.params.param('Parm3d').param("Search Date").value()
        search_datetime = parse(search_date)
        search_unix_time = search_datetime.timestamp()
        
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
        current_time_index = self.current_time_index
        current_unix_time = self.topo_unixtime_sorted[current_time_index]
        self.infl.setValue(current_unix_time)
        
        res_str = 'The result is ' + str(round(mn/60,2)) + 'mins from the search'
        self.params.param('Parm3d').param("Search Result").setValue(res_str)
        
    def previous_scene(self):
        """
        move to the previous scene
        """
        current_time_index = self.current_time_index
        n3dtimes = self.n3dtimes
        if current_time_index == 0:
            new_index = n3dtimes-1
        else:
            new_index = current_time_index - 1
        
        self.current_time_index = new_index
        self.load_3dmodel()
        current_time_index = self.current_time_index
        current_unix_time = self.topo_unixtime_sorted[current_time_index]
        self.infl.setValue(current_unix_time)
    
    def next_scene(self):
        """
        move to the next scene
        """
        current_time_index = self.current_time_index
        n3dtimes = self.n3dtimes
        if current_time_index == n3dtimes-1:
            new_index = 0
        else:
            new_index = current_time_index + 1
        
        self.current_time_index = new_index
        self.load_3dmodel()
        current_time_index = self.current_time_index
        current_unix_time = self.topo_unixtime_sorted[current_time_index]
        self.infl.setValue(current_unix_time)

class VizTopo(QtWidgets.QWidget):
    def __init__(self, topo_dictionary_list: list[dict], gl_option: str):
        """
        gui object for visualizing topologies
        """
        QtWidgets.QWidget.__init__(self)
        self.setupGUI()
        self.insert_parm3d()
        self.topo_dictionary_list: list[dict] = topo_dictionary_list
        """list of dictionary specifying the topologies to visualize"""
        self.gl_option: str = gl_option
        """gl option for the 3d view"""
        self.load_3dmodel(zoom_extent = True)
        self.backgroundc: str = 'k'
        """background color of the 3d view port"""
        
    def setupGUI(self):
        """
        function for setting up the base gui
        """
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
        self.splitter.setSizes([int(self.screen_width/3), self.screen_width])
        
    def insert_parm3d(self):
        """
        function for setting up the base parameters of the gui
        """
        self.export3d = dict(name='Export3d', type='group', expanded = True, title = "Export Image",
                    children=[
                                dict(name='xpixels', type = 'int', title = "XPixels", default = 1000, readonly = False),
                                dict(name='ypixels', type = 'int', title = "YPixels", default = 1000, readonly = False),
                                dict(name = 'Export', type = 'action')
                            ]
                    )

        self.params = Parameter.create(name = "Export", type = "group", children = [self.export3d])
        self.params.param('Export3d').param('Export').sigActivated.connect(self.export)
        self.tree.addParameters(self.params, showTop=False)
    
    def load_3dmodel(self, zoom_extent: bool = False):
        """
        loads the 3d model onto the 3d view port.

        Parameters
        ----------
        zoom_extent : bool, optional
            zoom to the 3d model. Default = False.
        """
        bbox_list = _convert_topo_dictionary_list4viz(self.topo_dictionary_list, self.view3d, gl_option=self.gl_option)
        if zoom_extent == True:
            overall_bbox = calculate.bbox_frm_bboxes(bbox_list)
            midpt = calculate.bboxes_centre([overall_bbox])[0]
            self.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
            
            lwr_left = [overall_bbox.minx, overall_bbox.miny, overall_bbox.minz]
            upr_right =  [overall_bbox.maxx, overall_bbox.maxy, overall_bbox.maxz]
            dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
            self.view3d.opts['distance'] = dist*1.5
            
        axs = gl.GLAxisItem()
        axs.setSize(x=1, y=1, z=1)
        self.view3d.addItem(axs)
        
    def export(self):
        """
        export the image on the 3d view port.
        """
        fn = pg.FileDialog.getSaveFileName(self, "Choose File Path", "exported_img.png", 'PNG (*.png);; JPEG (*.jpg)')
        
        if fn == '':
            return
        
        xpixels = self.params.param('Export3d').param('xpixels').value()
        ypixels = self.params.param('Export3d').param('ypixels').value()
        
        res_path = str(fn[0])
        _export_img(res_path, xpixels, ypixels, self.view3d)

def viz(topo_dictionary_list: list[dict], gl_option: str = 'opaque'):
    """
    This function visualises the topologies.
 
    Parameters
    ----------
    topo_dictionary_list : list[dict]
        A list of dictionary specifying the visualisation parameters.
        - topo_list: the topos to visualise
        - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        - draw_edges: bool whether to draw the edges of mesh, default is True
        - point_size: size of the point
        - px_mode: True or False, if true the size of the point is in pixel if not its in meters
        - attribute: name of the attribute to viz
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
    """
    pg.mkQApp()
    win = VizTopo(topo_dictionary_list, gl_option)
    win.setWindowTitle("Viz")
    win.show()
    win.resize(1100,700)

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()

def viz_animate(topo_2ddlist: list[list[dict]], topo_datetime_ls: list[datetime.datetime], gl_option: str = 'opaque'):
    """
    This function produces an animation window.
 
    Parameters
    ----------
    topo_2ddlist : list[list[dict]]
        A 2d list of dictionary specifying the visualisation parameters.
        - topo_list: the topos to visualise
        - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        - draw_edges: bool whether to draw the edges of mesh, default is True
        - point_size: size of the point
        - px_mode: True or False, if true the size of the point is in pixel if not its in meters
        - attribute: name of the attribute to viz
    
    topo_datetime_ls : list[datetime.datetime]
        timestamp of the topology data.
    
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
    """    
    #convert the datetimes to unix timestamp
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = dt.timestamp()
        unix43d.append(unix_time)
    
    pg.mkQApp()
    win = AnimateTopo(topo_2ddlist, unix43d, gl_option=gl_option)
    
    win.setWindowTitle("AnimateTopo")
    win.show()
    win.resize(1100,700)
    win.animate()

def viz_animate_falsecolour(topo_2dlist: list[list[topobj.Topology]], results2d: list[list[float]], topo_datetime_ls: list[datetime.datetime], 
                            false_min_max_val: list[float] = None, other_topo_2ddlist: list[list[dict]] = [], gl_option: str = 'opaque'):
    """
    This function produces a falsecolour animation window.
 
    Parameters
    ----------
    topo_2dlist : list[list[topobj.Topology]]
        2dlist of topologies. You can have one set of geometries to multiple sets of results.
    
    results2d : list[list[float]]
        2d list of performance value to be visualise as falsecolour.
    
    topo_datetime_ls : list[datetime.datetime]
        timestamp of the topology data.
    
    false_min_max_val : list[float], optional
        list of two values [min_val, max_val] for the falsecolour visualisation.
    
    other_topo_2ddlist : list[list[dict]], optional
        A 2d list of dictionary specifying the visualisation parameters.
        - topo_list: the topos to visualise
        - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        - draw_edges: bool whether to draw the edges of mesh, default is True
        - point_size: size of the point
        - px_mode: True or False, if true the size of the point is in pixel if not its in meters
        - attribute: name of the attribute to viz
    
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
    """
    
    #--------------------------------------------------------------------------------------------
    #convert the datetimes to unix timestamp
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = dt.timestamp()
        unix43d.append(unix_time)
    
    pg.mkQApp()
    win = AFalsecolour(topo_2dlist, results2d, unix43d, false_min_max_val = false_min_max_val, 
                       other_topo_2ddlist = other_topo_2ddlist, gl_option = gl_option)
    
    win.setWindowTitle("AnimateFalseColour")
    win.show()
    win.resize(1100,700)
    win.animate()

def viz_falsecolour(topo_list: list[topobj.Topology], results: list[float], false_min_max_val: list[float] = None, other_topo_dlist: list[dict] = [], 
                    gl_option: str = 'opaque'):
    """
    This function visualises the topologies in falsecolour.
 
    Parameters
    ----------
    topo_list : list[topobj.Topology]
        list of topologies.
    
    results : list[float]
        list of performance value to be visualise as falsecolour.
    
    min_max_val : list[float], optional
        list of two values [min_val, max_val] for the falsecolour visualisation.
    
    other_topo_dlist : list[dict], optional
        A list of dictionary specifying the visualisation parameters.
        - topo_list: the topos to visualise
        - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        - draw_edges: bool whether to draw the edges of mesh, default is True
        - point_size: size of the point
        - px_mode: True or False, if true the size of the point is in pixel if not its in meters
        - attribute: name of the attribute to viz
        
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
    """
    pg.mkQApp()
    win = FalsecolourView()
    win.setWindowTitle("FalseColourView")
    
    win.real_min_max = [min(results), max(results)]
    
    if false_min_max_val == None:
        colours, legendbar = win.gen_falsecolour_bar(win.real_min_max[0], win.real_min_max[1], results)
    else:
        colours, legendbar = win.gen_falsecolour_bar(false_min_max_val[0], false_min_max_val[1], results)
        
    win.insert_export_parm()
    colour_refs = win.colour_refs
    clr_ints = _sort_topos2clr_ints(topo_list, colours, colour_refs)
    
    for cnt,clr_int in enumerate(clr_ints):
        colour = colour_refs[cnt*2]
        _clr_topos(clr_int, colour, win.view3d, gl_option = gl_option)
    
    win.view3d.addItem(gl.GLAxisItem())
    cmp = create.composite(topo_list)
    bbox = calculate.bbox_frm_topo(cmp)
    midpt = calculate.bboxes_centre([bbox])[0]
    win.view3d.opts['center'] = QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
    
    lwr_left = [bbox.minx, bbox.miny, bbox.minz]
    upr_right =  [bbox.maxx, bbox.maxy, bbox.maxz]
    dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
    win.view3d.opts['distance'] = dist*1.5
    if len(other_topo_dlist) != 0:
        _convert_topo_dictionary_list4viz(other_topo_dlist, win.view3d, gl_option = gl_option)
    
    win.view3d.addItem(legendbar)
    
    win.show()
    win.resize(1100,700)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()

def viz_graph(xvalues2d: list[list], yvalues2d: list[list[float]], colour_ls: list[list[int]], xlabel: str = None, xunit: str = None, ylabel: str = None, 
              yunit: str = None, title: str = None, legend: list[str] = None, inf_lines: list[dict] = None, regions: list[dict] = None, 
              second_xvalues2d: list[list] = None, second_yvalues2d: list[list[float]] = None, second_colour_ls: list[list[int]] = None, 
              second_legend: list[str] = None, second_ylabel: str = None, second_yunit: str = None):
    """
    This function visualises time-series data.
 
    Parameters
    ----------
    xvalues2d : list[list] 
        2dlist of floats/datetime. xvalues to plot. You can input python datetime object for time-series data. For different set of data separate them into different list.
    
    yvalues2d : list[list[float]]
        yvalues to plot. For different set of data separate them into different list.
    
    colour_ls : list[list[int]]
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
        
    legend : list[str], optional
        list of strings for the legends.
        
    inf_lines : list[dict], optional
        dictionary describing the infinite line.
        - label: str to describe the infinite line
        - angle: float, the angle of the infinite line, 0=horizontal, 90=vertical
        - pos: float, for horizontal and vertical line a single value is sufficient, for slanted line (x,y) is required.
        - colour: tuple, (r,g,b,a)
    
    regions : list[dict], optional
        dictionary describing the region on the graph.
        - label: str to describe the region.
        - orientation: str, vertical or horizontal
        - range: list of floats, [lwr_limit, upr_limit] e.g. [50,70] 
        - colour: tuple, (r,g,b,a)
    
    second_xvalues2d : list[list]
        2dlist of floats/datetime. xvalues to plot on a second y-axis. You can input python datetime object for time-series data. For different set of data separate them into different list.
        
    second_yvalues2d : list[list[float]], optional
        yvalues to plot on a second y-axis. For different set of data separate them into different list.
        
    second_colour_ls : list[list[int]]
        list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)].
    
    second_legend : list[str], optional
        list of strings for the data on the second axis.
        
    second_ylabel : str, optional
        label for the second y-axis.
    
    second_yunit : str, optional
        unit label for the second y-axis.
    """
    
    #========================================================================
    pg.mkQApp()
    
    is_time_series = False
    
    if type(xvalues2d[0][0]) == datetime.datetime:    
        #convert datetime to unix timestamp
        is_time_series = True
        unix_2d = _convert_datetime2unix(xvalues2d)
        xvalues2d = unix_2d
        if second_xvalues2d is not None and second_yvalues2d is not None and second_colour_ls:
            unix2_2d = _convert_datetime2unix(second_xvalues2d)
            second_xvalues2d = unix2_2d
        
    win = GraphView(is_time_series)
    scatterplot_ls = _plot_graph(xvalues2d, yvalues2d, colour_ls)
    global p1, p2
    p1 = win.p1
    win.add_scatter(scatterplot_ls)
    p1.setTitle(title)
    p1.setLabel('bottom', text = xlabel, units = xunit)
    p1.setLabel('left', text = ylabel, units = yunit)
    
    legend_item = pg.LegendItem((80,60), offset=(70,20))
    legend_item.setParentItem(p1.graphicsItem())
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
                init_pos = init_pos.timestamp()
                
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
                rangex = [rangex[0].timestamp(), rangex[1].timestamp()]
                
            lr = pg.LinearRegionItem(values=rangex, orientation=orient, brush=colour, pen=[255,255,255,80], movable=False)
            label = pg.InfLineLabel(lr.lines[1], label, position=0.9, rotateAxis=(1,0), anchor=(1, 1))
            win.plot.addItem(lr)
            
        
    if second_xvalues2d is not None and second_yvalues2d is not None and second_colour_ls:
        #need to create a new second y-axis and link it
        ## create a new ViewBox, link the right axis to its coordinate system
        p2 = pg.ViewBox()
        p1.showAxis('right')
        p1.scene().addItem(p2)
        p1.getAxis('right').linkToView(p2)
        p2.setXLink(win.p1)
        p1.setLabel('right', text = second_ylabel, units = second_yunit)
        
        ## Handle view resizing 
        def updateViews():
            p2.setGeometry(p1.vb.sceneBoundingRect())
            p2.linkedViewChanged(p1.vb, p2.XAxis)
            
        updateViews()
        p1.vb.sigResized.connect(updateViews)
        
        scatter_ls2 = _plot_graph(second_xvalues2d, second_yvalues2d, second_colour_ls, symbol = 't')
        for scatter2 in scatter_ls2:
            p2.addItem(scatter2)
        
        if second_legend is None:
            for cnt, scatter2 in enumerate(scatter_ls2):
                legend_item.addItem(scatter2, 'legend' + str(cnt))
        else:
            for cnt, scatter2 in enumerate(scatter_ls2):
                legend_item.addItem(scatter2, second_legend[cnt])
                
    win.setWindowTitle("GraphView")
    win.show()
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()

def viz_st(topo_2dlist: list[list[topobj.Topology]], results2d: list[list[float]], topo_datetime_ls: list[datetime.datetime], 
           xvalues2d: list[list[datetime.datetime]], yvalues2d: list[list[float]], colour_ls: list[list[int]], false_min_max_val: list[float] = None, 
           other_topo_2ddlist: list[list[dict]] = [], xlabel: str = None, xunit: str = None, ylabel: str = None, yunit: str = None, title: str = None, 
           legend: list[str] = None, inf_lines: list[dict] = None, regions: list[dict] = None, second_xvalues2d: list[list[datetime.datetime]] = None, 
           second_yvalues2d: list[list[float]] = None, second_colour_ls: list[list[int]] = None, second_legend: list[str] = None, 
           second_ylabel: str = None, second_yunit: str = None, gl_option: str = 'opaque'):
    """
    This function visualises the spatial time-series data.
 
    Parameters
    ----------
    topo_2dlist : list[list[topobj.Topology]]
        2dlist of topologies. You can have one set of geometries to multiple sets of results.
    
    results2d : list[list[float]]
        2d list of performance value to be visualise as falsecolour.
    
    topo_datetime_ls : list[datetime.datetime]
        timestamp of the topology data.
    
    xvalues2d : list[list[datetime.datetime]]
        xvalues to plot. For different set of data separate them into different list.
    
    yvalues2d : list[list[float]]
        yvalues to plot. For different set of data separate them into different list.
    
    colour_ls : list[list[int]]
        list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)]. list[shape(number of colors, 3)] 
    
    false_min_max_val : list[float], optional
        list of two values [min_val, max_val] for the falsecolour visualisation.
    
    other_topo_2ddlist : list[list[dict]], optional
        A 2d list of dictionary specifying the visualisation parameters.
        - topo_list: the topos to visualise
        - colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        - draw_edges: bool whether to draw the edges of mesh, default is True
        - point_size: size of the point
        - px_mode: True or False, if true the size of the point is in pixel if not its in meters
        - attribute: name of the attribute to viz
    
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
    
    legend : list[str], optional
        list of strings for the legends.
        
    inf_lines : list[dict], optional
        dictionary describing the infinite line.
        - label: str to describe the infinite line
        - angle: float, the angle of the infinite line, 0=horizontal, 90=vertical
        - pos: float, for horizontal and vertical line a single value is sufficient, for slanted line (x,y) is required.
        - colour: tuple, (r,g,b,a)
        
    regions : list[dict], optional
        dictionary describing the region on the graph.
        - label: str to describe the region.
        - orientation: str, vertical or horizontal
        - range: list of floats, [lwr_limit, upr_limit] e.g. [50,70] 
        - colour: tuple, (r,g,b,a)
        
    second_xvalues2d : list[list[datetime.datetime]], optional
        xvalues to plot on a second y-axis. You can input python datetime object for time-series data. For different set of data separate them into different list.
        
    second_yvalues2d : list[list[float]], optional
        yvalues to plot on a second y-axis. For different set of data separate them into different list.
        
    second_colour_ls : list[list[int]], optional
        list of tuples specifying the colour. The tuple is (R,G,B,A). e.g. [(255,255,255,255), (255,0,0,255)].
    
    second_legend : list[str], optional
        list of strings for the data on the second axis.
    
    second_ylabel : str, optional
        label for the second y-axis.
    
    second_yunit : str, optional
        unit label for the second y-axis.
        
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
    """    
    #convert the datetimes to unix timestamp
    unix43d = []
    for dt in topo_datetime_ls:
        unix_time = dt.timestamp()
        unix43d.append(unix_time)
    
    unix_2d_graph = _convert_datetime2unix(xvalues2d)
    
    if second_xvalues2d is not None:
        second_xvalues2d  = _convert_datetime2unix(second_xvalues2d)
        
    pg.mkQApp()
    win = StView(topo_2dlist, results2d, unix43d, unix_2d_graph, yvalues2d, colour_ls, false_min_max_val = false_min_max_val, 
                 other_topo_2ddlist = other_topo_2ddlist, xlabel = xlabel, xunit = xunit, ylabel = ylabel, yunit = yunit, 
                 title = title, legend = legend, inf_lines = inf_lines, regions = regions, second_xvalues2d=second_xvalues2d,
                 second_yvalues2d=second_yvalues2d, second_colour_ls=second_colour_ls,second_legend=second_legend,
                 second_ylabel=second_ylabel, second_yunit=second_yunit, gl_option = gl_option)
    
    global p1, p2
    win.setWindowTitle("SpatialTimeSeriesView")
    win.showMaximized()
    win.animate()
            
def viz_vx_dict(vx_dict: dict, colour: str, wireframe: bool = True, gl_option: str = 'opaque'):
    """
    This function visualises the voxel dictionaries.
 
    Parameters
    ----------
    vx_dict : dict
        voxel dictionary generated from modify.xyzs2vox
        dictionary of voxels 
        vox_dim: the dimension of the voxels
        voxels: dictionary of voxels
    
    colour : str 
        can be a tuple too. keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        
    wireframe : bool, optional
        default True. 
    
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
    """
    vox_dim = vx_dict['voxel_dim']
    voxs = vx_dict['voxels']
    viz_ls = []
    for key in voxs.keys():
        vx = voxs[key]
        midpt = vx['midpt']
        box = create.box(vox_dim[0], vox_dim[1], vox_dim[2], centre_pt=midpt)
        if wireframe == False:
            viz_ls.append(box)
        else:
            bedges = get.edges_frm_solid(box)
            viz_ls.extend(bedges)
            
    viz([{'topo_list':viz_ls, 'colour':colour}], gl_option = gl_option)

def _clr_topos(topos: list[topobj.Topology], colour: list, view3d: gl.GLViewWidget, gl_option: str = 'opaque'):
    """
    This function colours the topology for falsecolour visualization.
 
    Parameters
    ----------
    topos : list[topobj.Topology]
        the topos to sort.
        
    colour : list
        [r,g,b]. The colour of the topologies.
        
    view3d: gl.GLViewWidget
        3d scene to put the objects into.
        
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
    """
    colour = np.append(colour, [255])
    colour = colour/255
    pts = []
    lines = []
    faces = []
    for topo in topos:
        topo_type = topo.topo_type
        if topo_type == topobj.TopoType.VERTEX:
            pnt = topo.point.xyz
            pts.append(pnt)
        elif topo_type == topobj.TopoType.EDGE:
            lines.append(topo)
        elif topo_type == topobj.TopoType.WIRE:
            edges = get.edges_frm_wire(topo)
            lines.extend(edges)
        elif topo_type == topobj.TopoType.FACE:
            faces.append(topo)
        elif topo_type == topobj.TopoType.SHELL:
            sfaces = get.faces_frm_shell(topo)
            faces.extend(sfaces)
        elif topo_type == topobj.TopoType.SOLID:
            sofaces = get.faces_frm_solid(topo)
            faces.extend(sofaces)
        elif topo_type == topobj.TopoType.COMPOSITE:
            sorted_d = get.unpack_composite(topo)
            vertices = sorted_d['vertex']
            if len(vertices) > 0:
                pnts = [v.point.xyz for v in vertices]
                pts.extend(pnts)
            #=================================================================================
            #get all the topology that can viz as lines
            #=================================================================================
            cedges = sorted_d['edge']
            if len(cedges) > 0:
                lines.extend(cedges)
            wires = sorted_d['wire']
            if len(wires) > 0:
                edges = np.array([get.edges_frm_wire(wire) for wire in wires], dtype=object)
                edges = list(chain(*edges))
                lines.extend(edges)
            #=================================================================================
            #get all the topology that can be viz as mesh
            #=================================================================================
            cfaces = sorted_d['face']
            if len(cfaces) > 0:
                faces.extend(cfaces)
            
            shells = sorted_d['shell']
            if len(shells) > 0:
                shells2faces = np.array([get.faces_frm_shell(shell) for shell in shells], dtype=object)
                shells2faces = list(chain(*shells2faces))
                faces.extend(shells2faces)
            
            solids = sorted_d['solid']
            if len(solids) > 0:
                solids2faces = np.array([get.faces_frm_solid(solid) for solid in solids], dtype=object)
                solids2faces = list(chain(*solids2faces))
                faces.extend(solids2faces)
    
    if len(pts) > 0:
        pts = np.array(pts)
        viz_pts = _make_points(pts, tuple(colour), 10, pxMode = True)
        view3d.addItem(viz_pts)
        
    if len(lines) > 0:
        line_vertices = modify.edges2lines(lines)
        viz_lines = _make_line(line_vertices, line_colour = tuple(colour))
        view3d.addItem(viz_lines)

    if len(faces) > 0:
        mesh_dict = modify.faces2trimesh(faces)
        #flip the indices
        verts = mesh_dict['vertices']
        idx = mesh_dict['indices']
        #flip the vertices to be clockwise
        idx = np.flip(idx, axis=1)
        viz_mesh = _make_mesh(verts, idx, draw_edges = False, gloptions = gl_option)
        viz_mesh.setColor(colour)
        view3d.addItem(viz_mesh)

def _convert_datetime2unix(datetime2d: list[list[datetime.datetime]]) -> list[list[float]]:
    """
    This function converts a 2d list of datetime object into unix timestamps.
 
    Parameters
    ----------
    datetime2d : list[list[datetime.datetime]]
        datetimes to be converted.
        
    Returns
    -------
    unix2d : list[list[float]]
        the converted unix timestamps.
    """
    unix2d = []
    for xvalues in datetime2d:
        unix_ls = []
        for x in xvalues:
            unix_time = x.timestamp()
            # dt = datetime.datetime.fromtimestamp(unix_time)
            # print(dt)
            unix_ls.append(unix_time)
        unix2d.append(unix_ls)
    return unix2d

def _convert_topo_dictionary_list4viz(topo_dictionary_list, view3d, gl_option='opaque'):
    """
    This function visualises the topologies.
 
    Parameters
    ----------
    topo_dictionary_list : a list of dictionary
        A list of dictionary specifying the visualisation parameters.
        topo_list: the topos to visualise
        colour:  keywords (red,orange,yellow,green,blue,black,white) or rgb tuple to specify the colours
        draw_edges: bool whether to draw the edges of mesh, default is True
        point_size: size of the point
        px_mode: True or False, if true the size of the point is in pixel if not its in meters
        gl_option: str, either 'opaque', 'additive' or 'translucent'. default is 'opaque'
        attribute: name of the attribute to visualise with the topologies
        
    view3d : pyqtgraph 3d view widget
        3d view to visualise the geometries
        
    gl_option : str, optional
        str describing the gl option for the 3d view, default is 'opqaue', can be 'opaque', 'additive', 'translucent' 
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
        
        return np.array([rgb])
    
    def viz_att(topo_list, att2viz):
        txts = []
        for topo in topo_list:
            att = topo.attributes
            if att2viz in att.keys():
                bbox = calculate.bbox_frm_topo(topo)
                pos = calculate.bboxes_centre([bbox])[0]
                txt = gl.GLTextItem(pos=pos, text=str(att[att2viz]))
                txts.append(txt)
                
        return txts
        
    bbox_list = []
    
    pv_ls = None
    line_ls = None
    mesh_v_ls = None
    mesh_i_ls = None
    mesh_line_ls = None
    
    pt_rgb_ls = None
    pt_size_ls = None
    px_mode_ls = None
    pt_indx_ls = []
    prev_npts = 0
    line_rgb_ls = None
    mesh_rgb_ls = None
    mesh_line_rgb_ls = None
    
    txt_ls = []
    for d in topo_dictionary_list:
        colour = d['colour']
        rgb = colour2rgb(colour)
        draw_edges = True
        if 'draw_edges' in d.keys():
            draw_edges = d['draw_edges']
        
        pt_size = [10]
        if 'point_size' in d.keys():
            pt_size = [d['point_size']]
            
        px_mode = True
        if 'px_mode' in d.keys():
            px_mode = [d['px_mode']]
        
        is_viz_att = False
        if 'attribute' in d.keys():
            is_viz_att = True
            att2viz = d['attribute']
        
        topo_list = d['topo_list']
        cmp = create.composite(topo_list)
        sorted_d = get.unpack_composite(cmp)
        #=================================================================================
        #get all the topology that can viz as points
        #=================================================================================
        vertices = sorted_d['vertex']
        if len(vertices) > 0:
            points_vertices = np.array([v.point.xyz for v in vertices])
            if pv_ls is None:
                pv_ls = points_vertices
            else:
                pv_ls = np.append(pv_ls, points_vertices, axis=0)
            
            npts = len(points_vertices)
            #----------------------------------------------------------------------------
            indx = [prev_npts, npts]
            prev_npts = npts
            pt_indx_ls.append(indx)
            #----------------------------------------------------------------------------
            #check the rgb
            rgb_pts = np.repeat(rgb, npts, axis=0)
            if pt_rgb_ls is None:
                pt_rgb_ls = rgb_pts
            else:
                pt_rgb_ls = np.append(pt_rgb_ls, rgb_pts, axis=0)
            #----------------------------------------------------------------------------
            #check the point size
            pt_sizes = np.repeat(pt_size, npts, axis=0)
            
            if pt_size_ls is None:
                pt_size_ls = pt_sizes
            else:
                pt_size_ls = np.append(pt_size_ls, pt_sizes, axis=0)    
            #----------------------------------------------------------------------------
            if px_mode_ls is None:
                px_mode_ls = np.array([px_mode])
            else:
                px_mode_ls = np.append(px_mode_ls, [px_mode], axis=0)
            #----------------------------------------------------------------------------
            # if attributes is true viz the attribute specify
            if is_viz_att == True:
                txts = viz_att(vertices, att2viz)
                txt_ls.extend(txts)
        #=================================================================================
        #get all the topology that can be viz as mesh
        #=================================================================================
        all_faces = []
        faces = sorted_d['face']
        if len(faces) > 0:
            all_faces = faces
            #----------------------------------------------------------------------------
            # if attributes is true viz the attribute specify
            if is_viz_att == True:
                txts = viz_att(faces, att2viz)
                txt_ls.extend(txts)
                
        shells = sorted_d['shell']
        if len(shells) > 0:
            shells2faces = np.array([get.faces_frm_shell(shell) for shell in shells], dtype=object)
            shells2faces = list(chain(*shells2faces)) #shells2faces.flatten()
            all_faces = np.append(all_faces, shells2faces)
            #----------------------------------------------------------------------------
            # if attributes is true viz the attribute specify
            if is_viz_att == True:
                txts = viz_att(shells, att2viz)
                txt_ls.extend(txts)
        
        solids = sorted_d['solid']
        if len(solids) > 0:
            solids2faces = np.array([get.faces_frm_solid(solid) for solid in solids], dtype=object)
            solids2faces = list(chain(*solids2faces))#solids2faces.flatten()
            all_faces = np.append(all_faces, solids2faces)
            #----------------------------------------------------------------------------
            # if attributes is true viz the attribute specify
            if is_viz_att == True:
                txts = viz_att(solids, att2viz)
                txt_ls.extend(txts)
        
        #if there are faces to be viz
        if len(all_faces) > 0:
            mesh_dict = modify.faces2trimesh(all_faces)
            #flip the indices
            verts = mesh_dict['vertices']
            idx = mesh_dict['indices']
            #flip the vertices to be clockwise
            idx = np.flip(idx, axis=1)
            #----------------------------------------------------------------------------
            if mesh_i_ls is None:
                mesh_i_ls = idx
            else:
                nv = len(mesh_v_ls)
                mesh_i_ls = np.append(mesh_i_ls, idx+nv, axis=0)
            #----------------------------------------------------------------------------
            if mesh_v_ls is None:
                mesh_v_ls = verts
            else:
                mesh_v_ls = np.append(mesh_v_ls, verts, axis=0)
            
            #----------------------------------------------------------------------------
            ntri = len(verts)
            mesh_rgbs = np.repeat(rgb, ntri, axis=0)
            if mesh_rgb_ls is None:
                mesh_rgb_ls = mesh_rgbs
            else:
                mesh_rgb_ls = np.append(mesh_rgb_ls, mesh_rgbs, axis=0)
            #----------------------------------------------------------------------------
            if draw_edges == True:
                #get all the boundary edges of the faces
                fcomp = create.composite(all_faces)
                fedges = get.edges_frm_composite(fcomp)
                fline_vertices = modify.edges2lines(fedges)
                if mesh_line_ls is None:
                    mesh_line_ls = fline_vertices
                else:
                    mesh_line_ls = np.append(mesh_line_ls, fline_vertices, axis=0)
                
                mline_rgbs = np.repeat([[1,1,1,1]], len(fline_vertices), axis=0)
                if mesh_line_rgb_ls is None:
                    mesh_line_rgb_ls = mline_rgbs
                else:
                    mesh_line_rgb_ls = np.append(mesh_line_rgb_ls, mline_rgbs, axis=0)
            
        #=================================================================================
        #get all the topology that can viz as lines
        #=================================================================================
        all_edges = []
        edges = sorted_d['edge']
        if len(edges) > 0:
            all_edges = edges
            #----------------------------------------------------------------------------
            # if attributes is true viz the attribute specify
            if is_viz_att == True:
                txts = viz_att(edges, att2viz)
                txt_ls.extend(txts)
        
        wires = sorted_d['wire']
        if len(wires) > 0:
            wires2edges = np.array([get.edges_frm_wire(wire) for wire in wires], dtype=object)
            wires2edges = list(chain(*wires2edges)) #wires2edges.flatten()
            all_edges = np.append(all_edges, wires2edges )
            # if attributes is true viz the attribute specify
            if is_viz_att == True:
                txts = viz_att(wires, att2viz)
                txt_ls.extend(txts)
        
        if len(all_edges) > 0:
            line_vertices = modify.edges2lines(all_edges)
            if line_ls is None:
                line_ls = line_vertices
            else:
                line_ls = np.append(line_ls, line_vertices, axis=0)                
            
            line_rgbs = np.repeat(rgb, len(line_vertices), axis=0)
            if line_rgb_ls is None:
                line_rgb_ls = line_rgbs
            else:
                line_rgb_ls = np.append(line_rgb_ls, line_rgbs, axis=0)
                
        #=================================================================================
        #find the bbox
        #=================================================================================
        bbox = calculate.bbox_frm_topo(cmp)
        bbox_list.append(bbox)
    
    #---------------------------------------------------------------------------------------------------
    # process the meshes
    if mesh_v_ls is not None:
        viz_mesh = _make_mesh(mesh_v_ls, mesh_i_ls, face_colours= mesh_rgb_ls, gloptions=gl_option ,draw_edges = False)
        view3d.addItem(viz_mesh)
        
    #---------------------------------------------------------------------------------------------------
    # process the lines
    if line_ls is not None:
        if mesh_line_ls is not None:
            line_ls = np.append(line_ls, mesh_line_ls, axis=0)
            line_rgb_ls = np.append(line_rgb_ls, mesh_line_rgb_ls, axis=0)
        
        viz_lines = _make_line(line_ls, line_colour = line_rgb_ls)
        view3d.addItem(viz_lines) 
    else:
        if mesh_line_ls is not None:
            line_ls = mesh_line_ls
            line_rgb_ls = mesh_line_rgb_ls
            viz_lines = _make_line(line_ls, line_colour = line_rgb_ls)
            view3d.addItem(viz_lines) 
    #---------------------------------------------------------------------------------------------------
    # process the points
    if pv_ls is not None:
     
        px_cond = np.where(px_mode_ls == True)[0]
        if len(px_cond) == len(px_mode_ls):
            #means all points are going to be viewed in the same mode
            viz_pts = _make_points(pv_ls, pt_rgb_ls, pt_size_ls, pxMode = px_mode_ls[0])
            view3d.addItem(viz_pts)
            
        else:
            for cnt,md in enumerate(px_mode_ls):
                pt_indx = pt_indx_ls[cnt]
                viz_pts = _make_points(pv_ls[pt_indx[0]:pt_indx[1]], 
                                        pt_rgb_ls[pt_indx[0]:pt_indx[1]], 
                                        pt_size_ls[pt_indx[0]:pt_indx[1]], 
                                        pxMode = md)
                view3d.addItem(viz_pts)
    #---------------------------------------------------------------------------------------------------
    # process the txt_ls
    for txt in txt_ls:
        view3d.addItem(txt)
    
    return bbox_list

def _export_img(res_path: str, xpixels: int, ypixels: int, view3d: gl.GLViewWidget):
    """
    Export the scene in the 3d view port as an image.
 
    Parameters
    ----------
    res_path : str
        path of the img file.
    
    xpixels : int
        number of pixels in the x direction.
    
    ypixels : int
        number of pixels in the y direction.

    view3d : gl.GLViewWidget
        the 3d view port.
    """
    # make the background image transparent
    view3d.setBackgroundColor([0,0,0,0])
    d = view3d.renderToArray((xpixels, ypixels))
    d = np.rot90(d)
    pg.makeQImage(d).save(res_path)
    view3d.setBackgroundColor('k')

def _make_mesh(xyzs: np.ndarray, indices: np.ndarray, face_colours: np.ndarray = [], shader: str = "shaded", gloptions: str = "opaque", 
               draw_edges: bool = False, edge_colours: list[float] = [0,0,0,1]) -> gl.GLMeshItem:
    """
    This function returns a Mesh Item that can be viz by pyqtgraph.
 
    Parameters
    ----------
    xyzs : np.ndarray 
        vertices of the mesh. np.ndarray(shape(Nvertices,3)).
        
    indices : np.ndarray
        indices of the mesh. np.ndarray(shape(Ntriangles,3))
        
    face_colours : np.ndarray, optional
        array of colours specifying the colours of each triangles of the mesh. np.ndarray(shape(Ntrianges, 4)).
        
    shader : str, optional
        specify the shader visualisation of the mesh. The options are: 
        shaded, balloon, normalColor, viewNormalColor, edgeHilight, 
        heightColor
        
    gloptions : str, optional
        specify the gloption visualisation of the mesh. The options are: 
        additive, opaque, translucent
    
    draw_edges : bool, optional
        toggle to whether draw the edges of the mesh
        
    edge_colours: list[float], option
        list with four elements specifying the RGB and Transparency of the edges.
        
    Returns
    -------
    mesh : gl.GLMeshItem
        mesh for visualisation.
    """
    if len(face_colours) == 0:
        mesh = gl.GLMeshItem(vertexes = xyzs, faces = indices, 
                             faceColors = None,
                             edgeColor = edge_colours,
                             smooth = True,
                             drawEdges = draw_edges, 
                             shader = shader,
                             glOptions = gloptions)
    else:
        mesh = gl.GLMeshItem(vertexes = xyzs, faces = indices,
                             faceColors = face_colours,
                             edgeColor = edge_colours,
                             smooth = True,
                             drawEdges = draw_edges, 
                             shader = shader,
                             glOptions = gloptions)
    
    return mesh

def _make_line(xyzs: np.ndarray, line_colour: list[float] = [0,0,0,1], width: float = 1, antialias: bool = True, mode: str = "lines") -> gl.GLLinePlotItem:
    """
    This function returns a Line Item that can be viz by pyqtgraph.
 
    Parameters
    ----------
    xyzs : np.ndarray
        vertices of the line. np.ndarray(shape(Nvertices,3))
         
    line_colour : list[float], optional
        also can be np.ndarray(shape(Nlines, 4)) array of colours specifying the colours of the line.
    
    width : float, optional
        the width of the line
        
    antialias: bool, option
        smooth line drawing
    
    mode: str, option
        - lines: Each pair of vertexes draws a single line segment.
        - line_strip: All vertexes are drawn as a continuous set of line segments
        
    Returns
    -------
    line : gl.GLLinePlotItem
        line for visualisation.
    """
    line = gl.GLLinePlotItem(pos = xyzs, color = line_colour, width = width, antialias = antialias, mode = mode)
    return line

def _make_points(xyzs: np.ndarray, point_colours: np.ndarray, sizes: float, pxMode: bool = True) -> gl.GLScatterPlotItem:
    """
    This function returns points Item that can be viz by pyqtgraph.
 
    Parameters
    ----------
    xyzs : np.ndarray 
        vertices of the points. np.ndarray(shape(Nvertices,3)).
        
    point_colour : np.ndarray
        np.ndarray(shape(number of points, 4) array of colours specifying the colours of the point, if a single tuple is given all points assumes that colour.
    
    sizes : float, optional
        list of floats of points sizes, if single val is given all point assumes that size
        
    pxMode: bool, option
        if True size is measured in pixels, if False size is based on units of the xyzs
        
    Returns
    -------
    points : gl.GLScatterPlotItem
        point for visualisation.
    """
    points = gl.GLScatterPlotItem(pos=xyzs, color=point_colours, size=sizes, pxMode = pxMode)
    return points

def _map_gen_bar(user_min: float, user_max: float, real_min: float, real_max: float, 
                 results: list[float]) -> tuple[np.ndarray, gl.GLGradientLegendItem, np.ndarray]:
    """
    This function sorts the normalized results into 10 equal intervals.
 
    Parameters
    ----------
    user_min : float
        the minimum of the falsecolor bar.

    user_max : float
        the maximum of the falsecolor bar.

    real_min : float
        the real minimum of the results.

    real_max : float
        the real maximum of the results.

    results : list[float]
        the result values.
    
    Returns
    -------
    colors : np.ndarray
        array of floats between 0-1.
    
    gll : gl.GLGradientLegendItem
        gradient object for inserting into the 3d view port.

    bcolour : np.ndarray
        array of floats describing the falsecolor bar. np.ndarray(shape(10,3))
    """
    bcolour = np.array([[0,0,0.5], [0,0,0.5],  
                        [0,1,1], [0,1,1],
                        [0,0.5,0], [0,0.5,0],
                        [0,1,0], [0,1,0], 
                        [1,1,0], [1,1,0],
                        [1,0.59,0], [1,0.59,0], 
                        [1,0,1], [1,0,1], 
                        [0.42,0.33,0.33], [0.42,0.33,0.33],
                        [0.5,0,0], [0.5,0,0], 
                        [1,0,0], [1,0,0]])
    
    bcolour = bcolour*255
    
    pos = [0, 0.099, 
           0.1, 0.199,
           0.2, 0.299,
           0.3, 0.399,
           0.4, 0.499,
           0.5, 0.599,
           0.6, 0.699,
           0.7, 0.799,
           0.8, 0.899,
           0.9, 1.0,]

    color_map = pg.ColorMap(pos, bcolour)
    
    r = (np.array(results) - user_min)/ (user_max - user_min)
    new_r = _res2interval(r)
    
    colours = color_map.map(new_r, mode='float').round(2)
    colours = colours*255
    
    legendLabels = np.round(np.linspace(user_max, user_min, 11), decimals = 2)
    legendLabels[0] = real_max
    legendLabels[-1] = real_min
    legendPos = np.linspace(1.0, 0.0, 11)
    legend = dict(zip(map(str, legendLabels), legendPos))
    
    gll = gl.GLGradientLegendItem(pos=(10, 10), size=(50, 300), gradient=color_map, labels=legend)
    
    return colours, gll, bcolour

def _plot_graph(xvalues2d: list[list[float]], yvalues2d: list[list[float]], colour_ls: list[list[int]], symbol: str = 'o') -> list[pg.ScatterPlotItem]:
    """
    This function converts datasets to pg.ScatterPlotItem to be plotted on the gui.
 
    Parameters
    ----------
    xvalues2d : list[list[float]]
        xvalues to be plotted.
    
    yvalues2d : list[list[float]]
        yvalues to be plotted.
    
    colour_ls : list[list[int]]
        colors of each point. list[shape(number of points, 3)]

    xvalues2d : list[list[float]]
        xvalues to be plotted.
        
    Returns
    -------
    scatterplot_ls : list[pg.ScatterPlotItem]
        list of pg.ScatterPlotItem to be plotted.
    """
    scatterplot_ls = []
    for cnt, xvalues in enumerate(xvalues2d):
        yvalues = yvalues2d[cnt]
        all_vals = [xvalues, yvalues]
        zip_vals = list(zip(*all_vals))
        spots = []
        for zval in zip_vals:
            spots.append({'pos': zval, 'size':10, 'symbol':symbol})
        
        scatterplot = pg.ScatterPlotItem(size = 10, hoverable = True, 
                                         brush= pg.mkBrush(colour_ls[cnt][0], colour_ls[cnt][1],colour_ls[cnt][2], colour_ls[cnt][3]), 
                                         hoverSymbol='s', hoverPen=pg.mkPen('r', width=2))
        scatterplot.addPoints(spots)
        scatterplot_ls.append(scatterplot)
    return scatterplot_ls

def _res2interval(results: list[float]) -> np.ndarray:
    """
    This function sorts the normalized results into 10 equal intervals.
 
    Parameters
    ----------
    results : list[float]
        list or array of floats between 0-1.
    
    Returns
    -------
    new_res : np.ndarray
        array of floats between 0-1.
    """
    new_res = results[:]
    for i in range(10):
        int_mn = i*0.1
        int_mx = (i+1)*0.1
        condition = np.logical_and(int_mn <= new_res, new_res < int_mx)
        new_res = np.where(condition, int_mn, new_res)
    return new_res

def _sort_topos2clr_ints(topo_list: list[topobj.Topology], colour_list: list[list[int]], colour_refs: list[list[int]]):
    """
    This function sorts the topology into intervals for falsecolour visualization.
 
    Parameters
    ----------
    topo_list : list[topobj.Topology]
        the topos to sort.
        
    colour_list : list[list[int]]
        list of colours [[r,g,b,a],[r,g,b,a]]. The colours corresponding to the topologies. 
        
    colour_refs : list[list[int]]
        list of colours [[r,g,b,a],[r,g,b,a]]. The colours of the colourbar
    """
    clr_ints = []
    for i in range(10):
        colour = colour_refs[i*2]
        colour = np.append(colour, 255)
        colours_t = colour_list.T

        cond1 = np.where(colours_t[0] == colour[0], True, False)
        cond2 = np.where(colours_t[1] == colour[1], True, False)
        cond3 = np.where(colours_t[2] == colour[2], True, False)

        cond = np.logical_and(cond1, cond2)
        cond = np.logical_and(cond, cond3)

        idx = np.where(cond)
        chosen = np.take(topo_list, idx, axis=0)[0]
        clr_ints.append(chosen)
        
    return clr_ints

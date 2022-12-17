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
import colorsys
from itertools import chain

import numpy as np

from . import modify
from . import get
from . import create
from . import calculate
from . import topobj

class CoordinateSystem(object):
    """
    A coordinate system object
    
    Parameters
    ----------
    origin : tuple
        The xyz defining the origin.
        
    x_dir : tuple
        The xyz of a vector defining the x-axis
        
    y_dir : tuple
        The xyz of a vector defining the y-axis  
    
    Attributes
    ----------    
    origin : tuple
        The xyz defining the origin.
        
    x_dir : tuple
        The xyz of a vector defining the x-axis.
        
    y_dir : tuple
        The xyz of a vector defining the y-axis.
    """
    def __init__(self, origin, x_dir, y_dir):
        """Initialises the class"""
        if type(origin) != np.ndarray:
            origin = np.array(origin)
        if type(x_dir) != np.ndarray:
            x_dir = np.array(x_dir)
        if type(y_dir) != np.ndarray:
            y_dir = np.array(y_dir)
            
        self.origin = origin
        self.x_dir = x_dir
        self.y_dir = y_dir

class Ray(object):
    """
    A ray object
    
    Parameters
    ----------
    origin : tuple
        The xyz defining the origin.
        
    dirx : tuple
        The direction of the ray
    
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Attributes
    ----------    
    origin : tuple
        The xyz defining the origin.
        
    dirx : tuple
        The direction of the ray
        
    """
    def __init__(self, origin, dirx, attributes = {}):
        """Initialises the class"""
        if type(origin) != np.ndarray:
            origin = np.array(origin)
        if type(dirx) != np.ndarray:
            dirx = np.array(dirx)
            
        self.origin = origin
        self.dirx = dirx
        self.attributes = attributes
    
    def overwrite_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att 

class Bbox(object):
    """
    A bounding box object
    
    Parameters
    ----------
    bbox_arr : tuple
        Array specifying [minx,miny,minz,maxx,maxy,maxz].
        
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Attributes
    ----------    
    minx : float
        The min x.
    
    miny : float
        The min y.
        
    minz : float
        The min z.
        
    maxx : float
        The max x.
        
    maxy : float
        The max y.
        
    maxz : float
        The max z.
        
    attributes : dictionary
        dictionary of the attributes
        
    """
    def __init__(self, bbox_arr, attributes = {}):
        """Initialises the class"""
        if type(bbox_arr) != np.ndarray:
            bbox_arr = np.array(bbox_arr)
        
        self.bbox_arr = bbox_arr
        self.minx = bbox_arr[0]
        self.miny = bbox_arr[1]
        self.minz = bbox_arr[2]
        self.maxx = bbox_arr[3]
        self.maxy = bbox_arr[4]
        self.maxz = bbox_arr[5]
        self.attributes = attributes
    
    def overwrite_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att

def id_dup_indices_1dlist(lst):
    """
    This function returns numpy array of the indices of the repeating elements in a list.
 
    Parameters
    ----------
    lst : a 1D list
        A 1D list to be analysed.
 
    Returns
    -------
    indices : 2d ndarray
        ndarray of the indices (Nduplicates, indices of each duplicate).
    """
    if type(lst) != np.ndarray:    
        lst = np.array(lst)
        
    idx_sort = np.argsort(lst)
    sorted_lst = lst[idx_sort]
    
    vals, idx_start, count = np.unique(sorted_lst, 
                                       return_counts=True,
                                       return_index=True)

    res = np.split(idx_sort, idx_start[1:])
    vals = vals[count > 1]
    res = list(filter(lambda x: x.size > 1, res))
    return res

def find_xs_not_in_ys(xlst, ylst):
    """
    This function compare the 2 list and find the elements in lst2find that is not in ref_lst.
 
    Parameters
    ----------
    xlst : a 1D list
        The lst to check if the elements exist in the ylst.
    
    ylstt : a 1D list
        The reference lst to compare xlst.
        
    Returns
    -------
    not_in : 1d ndarray
        the elements in xlst and not in ylst.
    """
    not_in_true = np.in1d(xlst, ylst)
    not_in_true = np.logical_not(not_in_true)
    not_in_indx = np.where(not_in_true)[0]
    not_in = np.take(xlst, not_in_indx, axis=0)
    return not_in
    
def gen_gridxyz(xrange, yrange, zrange=None):
    """
    This function generate a 2d xy grid.
    
    Parameters
    ----------
    xrange : list of int
        [start, stop, number of intervals].
    
    yrange : list of int
        [start, stop, number of intervals].
    
    zrange : list of int, optional
        [start, stop, number of intervals].
        
    Returns
    -------
    gridxy : 2d array
        2d array with xyz. If zrange == None, xy.
    """
    x = np.linspace(xrange[0], xrange[1], xrange[2])
    y = np.linspace(yrange[0], yrange[1], yrange[2])
    
    if zrange == None:
        xx, yy = np.meshgrid(x,y)
        xx = xx.flatten()
        yy= yy.flatten()
        xys = np.array([xx, yy])
        xys = xys.T
        return xys
    else:
        z = np.linspace(zrange[0], zrange[1], zrange[2])
        xx, yy, zz = np.meshgrid(x,y,z)
        xx = xx.flatten()
        yy= yy.flatten()
        zz = zz.flatten()
        xyzs = np.array([xx, yy, zz])
        xyzs = xyzs.T
        return xyzs

def convert_topo_dictionary_list4viz(topo_dictionary_list, view3d):
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
            viz_pts = make_points(points_vertices, rgb, pt_size, pxMode = px_mode)
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
            viz_lines = make_line(line_vertices, line_colour = rgb)
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
            viz_mesh = make_mesh(verts, idx,  draw_edges = draw_edges)
            viz_mesh.setColor(np.array(rgb))
            view3d.addItem(viz_mesh)     
        #=================================================================================
        #find the bbox
        #=================================================================================
        bbox = calculate.bbox_frm_topo(cmp)
        bbox_list.append(bbox)
    
    return bbox_list

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
        att: name of the att to visualise with the topologies
    
    """
    import pyqtgraph as pg
    from pyqtgraph.parametertree import Parameter, ParameterTree
    from pyqtgraph.Qt import QtCore, QtWidgets, QtGui
    import pyqtgraph.opengl as gl
    
    def sort_topos(topo, topo_type, icnt, topo_int_ls):
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
    
    def create_topo_int_ls(int_ls):
        topo_int_ls = []
        for _ in range(len(int_ls)):
            topo_int_ls.append([])
            topo_int_ls[-1].append([])
            topo_int_ls[-1].append([])
            topo_int_ls[-1].append([])
        return topo_int_ls
    
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
            # self.splitter.setStretchFactor(0,3)
            #put the splitter 2 into the layout
            self.layout.addWidget(self.splitter)
            
            self.view3d = gl.GLViewWidget()
            #self.view3d.opts['distance'] = 10000
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
    # import PyQt6
    os.environ['PYQTGRAPH_QT_LIB'] = "PyQt6"
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    pg.mkQApp()
    win = FalsecolourView()
    win.setWindowTitle("FalseColourView")
    if false_min_max_val == None:
        int_ls, c_ls = win.insert_fcolour(min(results), max(results), results)
    else:
        int_ls, c_ls = win.insert_fcolour(false_min_max_val[0], 
                                          false_min_max_val[1], results)

    topo_int_ls = create_topo_int_ls(int_ls)
    # print(len(topo_list))
    #sort all the topologies into the 10 intervals
    for cnt,topo in enumerate(topo_list):
        res = results[cnt]
        topo_type = topo.topo_type
        for icnt, intx in enumerate(int_ls):
            if icnt == 0:
                if res < intx[1]:
                    sort_topos(topo, topo_type, icnt, topo_int_ls)
                    break
                
            elif icnt == len(int_ls)-1:
                if res >= intx[0]:
                    sort_topos(topo, topo_type, icnt, topo_int_ls)
                    break
                
            else:
                if intx[0] <= res < intx[1]:
                    sort_topos(topo, topo_type, icnt, topo_int_ls)
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
            viz_pts = make_points(all_pts, rgb, 10, pxMode = True)
            win.view3d.addItem(viz_pts)
        if len(all_edges) > 0:
            line_vertices = modify.edges2lines(all_edges)
            viz_lines = make_line(line_vertices, line_colour = rgb)
            win.view3d.addItem(viz_lines)
        if len(all_faces) > 0:
            mesh_dict = modify.faces2mesh(all_faces)
            #flip the indices
            verts = mesh_dict['vertices']
            idx = mesh_dict['indices']
            #flip the vertices to be clockwise
            idx = np.flip(idx, axis=1)
            viz_mesh = make_mesh(verts, idx, 
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
        convert_topo_dictionary_list4viz(other_topo_dlist, win.view3d)
    
    win.show()
    win.resize(1100,700)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        pg.exec()
        
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
        att: name of the att to visualise with the topologies
    """
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph.Qt import QtCore, QtGui
    # import PyQt6
    os.environ['PYQTGRAPH_QT_LIB'] = "PyQt6"
    # Create a GL View widget to display data
    QtCore.QCoreApplication.setAttribute(QtCore.Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    pg.mkQApp('')
    w = gl.GLViewWidget()
    w.clear()
    bbox_list = convert_topo_dictionary_list4viz(topo_dictionary_list, w)
    
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
    
def make_mesh(xyzs, indices, face_colours = [], shader = "shaded", 
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

    import pyqtgraph.opengl as gl
    
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

def make_line(xyzs, line_colour = (0,0,0,1), width = 1, antialias=True, mode="lines"):
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

    import pyqtgraph.opengl as gl
    line = gl.GLLinePlotItem(pos=xyzs, color= line_colour, width=width, antialias=antialias, mode = mode)
    return line

def make_points(xyzs, point_colours, sizes, pxMode = True):
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
    import pyqtgraph.opengl as gl
    points = gl.GLScatterPlotItem(pos=xyzs, color=point_colours, size=sizes, pxMode = pxMode)
    return points

def pseudocolor(val, minval, maxval, inverse = False):
    """
    This function converts a value into a rgb value with reference to the minimum and maximum value.
 
    Parameters
    ----------
    val : float
        The value to be converted into rgb.
        
    minval : float
        The minimum value of the falsecolour rgb.
        
    maxval : float
        The maximum value of the falsecolour rgb.
    
    inverse : bool
        False for red being max, True for blue being maximum.
        
    Returns
    -------
    rgb value : tuple of floats
        The converted rgb value.
    """
    # convert val in range minval..maxval to the range 0..120 degrees which
    # correspond to the colors red..green in the HSV colorspace
    if val <= minval:
        if inverse == False:
            h = 250.0
        else:
            h=0.0
    elif val>=maxval:
        if inverse == False:
            h = 0.0
        else:
            h=250.0
    else:
        if inverse == False:
            h = 250 - (((float(val-minval)) / (float(maxval-minval)))*250)
        else:
            h = (((float(val-minval)) / (float(maxval-minval)))*250)
    # convert hsv color (h,1,1) to its rgb equivalent
    # note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
    return r, g, b
    
def calc_falsecolour(vals, minval, maxval, inverse = False):
    """
    This function converts a list of values into a list of rgb values with reference to the minimum and maximum value.
 
    Parameters
    ----------
    vals : list of float
        A list of values to be converted into rgb.
        
    minval : float
        The minimum value of the falsecolour rgb.
        
    maxval : float
        The maximum value of the falsecolour rgb.
    
    inverse : bool
        False for red being max, True for blue being maximum.
        
    Returns
    -------
    rgb value : list of tuple of floats
        The converted list of rgb value.
    """
    res_colours = []
    for result in vals:
        r,g,b = pseudocolor(result, minval, maxval, inverse=inverse)
        colour = (r, g, b)
        res_colours.append(colour)
    return res_colours

def rgb2val(rgb, minval, maxval):
    """
    This function converts a rgb of value into its original value with reference to the minimum and maximum value.
 
    Parameters
    ----------
    rgb : tuple of floats
        The rgb value to be converted.
        
    minval : float
        The minimum value of the falsecolour rgb.
        
    maxval : float
        The maximum value of the falsecolour rgb.
        
    Returns
    -------
    original value : float
        The orignal float value.
    """
    hsv = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])
    y = hsv[0]*360
    orig_val_part1 = ((-1*y) + 250)/250.0
    orig_val_part2 = maxval-minval
    orig_val = (orig_val_part1*orig_val_part2)+minval
    return orig_val

def write2ply(topo_list, ply_path, square_face = False):
    """
    Writes the topologies to a ply file. only works for points and face
 
    Parameters
    ----------
    topo_list : list of topo objects
        Topos to be written to ply.
        
    ply_path : str
        Path to write to.
        
    square_face : bool, optional
        If you know all the faces have 4 vertices and you want to preserve that turn this to True.
        
    """
    header = ['ply\n', 'format ascii 1.0\n', 'element vertex', 
              'property float32 x\n', 'property float32 y\n', 
              'property float32 z\n', 'element face',
              'property list uint8 int32 vertex_index\n', 'end_header\n']
    
    cmp = create.composite(topo_list)
    sorted_d = get.unpack_composite(cmp)
    #=================================================================================
    #get all the topology that can be viz as mesh
    #=================================================================================
    all_faces = []
    faces = sorted_d['face']
    if len(faces) > 0:
        all_faces = faces
    
    shells = sorted_d['shell']
    if len(shells) > 0:
        shells2faces = np.array([get.faces_frm_shell(shell) for shell in shells])
        shells2faces = shells2faces.flatten()
        all_faces = np.append(all_faces, shells2faces)
    
    solids = sorted_d['solid']
    if len(solids) > 0:
        solids2faces = np.array([get.faces_frm_solid(solid) for solid in solids])
        solids2faces = solids2faces.flatten()
        all_faces = np.append(all_faces, solids2faces)
    
    nverts = 0
    verts = []
    idxs = []
    #if there are faces to be viz
    if len(all_faces) > 0:
        if square_face == True:
            f_v = []
            for f in all_faces:
                f_v.extend(get.vertices_frm_face(f).tolist())

            verts = [v.point.xyz.tolist() for v in f_v]
            idxs = np.arange(len(verts))
            idxs = np.reshape(idxs, [len(all_faces), 4])
        else:
            mesh_dict = modify.faces2mesh(all_faces)
            verts = mesh_dict['vertices']
            idxs = mesh_dict['indices']
            #flip the vertices to be clockwise
            idxs = np.flip(idxs, axis=1)
            
    #=================================================================================
    #get all the topology that can viz as points
    #=================================================================================
    vertices = sorted_d['vertex']
    if len(vertices) > 0:
        points_vertices = [v.point.xyz.tolist() for v in vertices]
        verts.extend(points_vertices)
        
    nverts = len(verts)
    header[2] = 'element vertex ' + str(nverts) + '\n'
    nfaces = len(idxs)
    header[6] = 'element face ' + str(nfaces) + '\n'

    for xyz in verts:
        v_str = str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + '\n'
        header.append(v_str)
    if square_face == True:
        for idx in idxs:
            i_str = '4 ' + str(idx[0]) + ' ' + str(idx[1]) + ' ' + str(idx[2]) + ' ' + str(idx[3]) + '\n'
            header.append(i_str)
    else:
        for idx in idxs:
            i_str = '3 ' + str(idx[0]) + ' ' + str(idx[1]) + ' ' + str(idx[2]) + '\n'
            header.append(i_str)
        
    f = open(ply_path, "w")
    f.writelines(header)
    f.close()
    
def write2pts(vertex_list, pts_path):
    """
    Writes the vetices to a pts file. only works for vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology. 
        
    pts_path : str
        Path to write to.
        
    """
    xyz_ls = []
    for v in vertex_list:
        xyz = v.point.xyz
        normal = [0,0,0]
        att = v.attributes
        if 'normal' in att:
            normal = att['normal']
        v_str = str(xyz[0]) + ',' + str(xyz[1]) + ',' + str(xyz[2]) + ',' + str(normal[0]) + ',' + str(normal[1]) + ',' + str(normal[2]) + '\n'
        xyz_ls.append(v_str)
    
    f = open(pts_path, "w")
    f.writelines(xyz_ls)
    f.close()
    
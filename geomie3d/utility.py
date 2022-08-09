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

import numpy as np

import pyqtgraph as pg
from pyqtgraph.parametertree import Parameter, ParameterTree
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph.opengl as gl
import colorsys

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

class FalsecolourView(QtGui.QWidget):
    def __init__(self):
        QtGui.QWidget.__init__(self)
        self.setupGUI()
        
    def setupGUI(self):
        self.layout = QtGui.QVBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        #create the right panel
        self.splitter = QtGui.QSplitter()
        self.splitter.setOrientation(QtCore.Qt.Horizontal)
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
        bcolour = calc_falsecolour(float_list, min_val, max_val)
        new_c_list = []
        for c in bcolour:
            new_c = [c[0]*255, c[1]*255, c[2]*255]
            new_c_list.append(new_c)
            
        rangex = max_val-min_val
        intervals = rangex/10.0
        intervals_half = intervals/2.0
        interval_ls = []
        str_list = []
        fcnt = 0
        for f in float_list:
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
            fcnt+=1
            
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
        
        topo_list = d['topo_list']
        cmp = create.composite(topo_list)
        sorted_d = get.unpack_composite(cmp)
        #=================================================================================
        #get all the topology that can viz as points
        #=================================================================================
        vertices = sorted_d['vertex']
        if len(vertices) > 0:
            points_vertices = np.array([v.point.xyz for v in vertices])
            viz_pts = make_points(points_vertices, rgb, 10, pxMode = True)
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
            wires2edges = np.array([get.edges_frm_wire(wire) for wire in wires])
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
            shells2faces = np.array([get.faces_frm_shell(shell) for shell in shells])
            shells2faces = shells2faces.flatten()
            all_faces = np.append(all_faces, shells2faces)
        
        solids = sorted_d['solid']
        if len(solids) > 0:
            solids2faces = np.array([get.faces_frm_solid(solid) for solid in solids])
            solids2faces = solids2faces.flatten()
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
        att: name of the att to visualise with the topologies
    
    """
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
            faces = get.faces_frm_shell(topo)
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
    #========================================================================
    import PyQt5
    os.environ['PYQTGRAPH_QT_LIB'] = "PyQt5"
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
    win.view3d.opts['center'] = PyQt5.QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
    
    lwr_left = [bbox[0], bbox[1], bbox[2]]
    upr_right =  [bbox[3], bbox[4], bbox[5]]
    dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
    win.view3d.opts['distance'] = dist*1.5
    if len(other_topo_dlist) != 0:
        convert_topo_dictionary_list4viz(other_topo_dlist, win.view3d)
    
    win.show()
    win.resize(1100,700)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
        
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
        att: name of the att to visualise with the topologies
    """
    import PyQt5
    os.environ['PYQTGRAPH_QT_LIB'] = "PyQt5"
    # Create a GL View widget to display data
    QtGui.QApplication([])
    w = gl.GLViewWidget()
    
    bbox_list = convert_topo_dictionary_list4viz(topo_dictionary_list, w)
    
    w.addItem(gl.GLAxisItem())
    overall_bbox = calculate.bbox_frm_bboxes(bbox_list)
    midpt = calculate.bbox_centre(overall_bbox)
    w.opts['center'] = PyQt5.QtGui.QVector3D(midpt[0], midpt[1], midpt[2])
    
    lwr_left = [overall_bbox[0], overall_bbox[1], overall_bbox[2]]
    upr_right =  [overall_bbox[3], overall_bbox[4], overall_bbox[5]]
    dist = calculate.dist_btw_xyzs(lwr_left, upr_right)
    w.opts['distance'] = dist*1.5
    
    w.show()
    w.setWindowTitle('Geomie3D viz')
    QtGui.QApplication.instance().exec_()
        
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

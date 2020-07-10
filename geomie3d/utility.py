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

import numpy as np

from pyqtgraph.Qt import QtGui
import pyqtgraph.opengl as gl

from . import modify

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
    return np.array(res)
    
def viz(topo_dictionary_list):
    """
    This function visualises the topologies.
 
    Parameters
    ----------
    topo_dictionary_list : a list of dictionary
        A list of dictionary specifying the visualisation parameters.
        topo_list: the list of topos to visualise
        colour:  keywords (RED,ORANGE,YELLOW,GREEN,BLUE,BLACK,WHITE) or rgb tuple to specify the colours
        att: name of the att to visualise with the topologies
    """
    os.environ['PYQTGRAPH_QT_LIB'] = "PyQt5"
    ## Create a GL View widget to display data
    app = QtGui.QApplication([])
    w = gl.GLViewWidget()
    w.show()
    w.setWindowTitle('Geomie3D viz')
    topo2dlist = np.array([d["topo_list"] for d in topo_dictionary_list])
    topo_list = topo2dlist.flatten()
    #TODO figure out how to viz the topo
    mesh_dict_list = [modify.face2mesh(topo) for topo in topo_list]
    mesh_list = [make_mesh(md["vertices"], md["indices"], draw_edges = False) 
                 for md in mesh_dict_list]

    for mesh in mesh_list:
        mesh.setColor(np.array([0,0,1,1]))
        w.addItem(mesh)
        
    w.setCameraPosition(distance=60)
    QtGui.QApplication.instance().exec_()
        
def make_mesh(vertices, indices, face_colours = [], shader = "shaded", 
              gloptions = "opaque", draw_edges = False, 
              edge_colours = [0,0,0,1]):
    """
    This function returns a Mesh Item that can be viz by pyqtgraph.
 
    Parameters
    ----------
    vertices : ndarray of shape(Nvertices,3)
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
        mesh = gl.GLMeshItem(vertexes = vertices, faces = indices, 
                             faceColors = None,
                             edgeColor = edge_colours,
                             smooth = False,
                             drawEdges = draw_edges, 
                             # shader = shader,
                             glOptions = gloptions)
    else:
        mesh = gl.GLMeshItem(vertexes = vertices, faces = indices,
                             faceColors = face_colours,
                             edgeColor = edge_colours,
                             smooth = False,
                             drawEdges = draw_edges, 
                             # shader = shader,
                             glOptions = gloptions)
    
    return mesh
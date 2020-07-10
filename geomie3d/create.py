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
import numpy as np

from . import geom
from . import topo

def vertex(xyz, attributes = {}):
    """
    This function constructs a vertex topology.
 
    Parameters
    ----------
    xyz : tuple
        tuple with the xyz coordinates.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    vertex : vertex topology
        A vertex topology containing a point geometry
    """
    point = geom.Point(xyz)
    vertex = topo.Vertex(point, attributes = attributes)
    return vertex
    
def vertex_list(xyz_list, attributes_list = []):
    """
    This function constructs a list of vertex topology.
 
    Parameters
    ----------
    xyz_list : list of tuple
        list of tuple with the xyz coordinates.
        
    attributes_list : list of dictionary, optional
        Dictionary of the attributes
 
    Returns
    -------
    vertex_list : list of vertex topology
        A list of vertex topology.
    """
    nxyz = len(xyz_list)
    natts = len(attributes_list)
    
    #check if the list match
    is_atts = False
    if natts != 0:
        is_atts = True
        if natts != nxyz:
            raise NameError("Number of xyz_list and attributes_list do not match")
    
    if is_atts:
        vlist = np.array([vertex(xyz_list[cnt], attributes = attributes_list[cnt]) 
                          for cnt in range(nxyz)])
    else:
        vlist = np.array([vertex(xyz_list[cnt]) for cnt in range(nxyz)])
    
    return vlist

def polygon_face_frm_verts(vertex_list, attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    pass

def polygon_face_frm_wires(bdry_wire, hole_wire_list = [], attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    face = topo.Face(attributes = attributes)
    face.add_polygon_surface(bdry_wire, hole_wire_list = hole_wire_list)
    return face

def pline_edge_frm_verts(vertex_list, attributes = {}):
    """
    This function constructs a polyline edge from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    pline_edge : edge topology
        A edge topology containing a polyline geometry
    """
    edge = topo.Edge(attributes = attributes)
    edge.add_polyline_curve(vertex_list)
    return edge

def wire_frm_edges(edge_list, attributes = {}):
    """
    This function constructs a wire from a list of edges.
 
    Parameters
    ----------
    edge_list : a list of edge
        A list of edge topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    wire : wire topology
        A wire topology 
    """
    wire = topo.Wire(edge_list, attributes = attributes)
    return wire
    
def coordinate_system(origin, x_dir, y_dir):
    """
    This function creates a coordinate system object.
 
    Parameters
    ----------
    origin : tuple
        xyz coordinate that defines the origin.
    
    x_dir : tuple
        The xyz of a vector defining the x-axis
        
    y_dir : tuple
        The xyz of a vector defining the y-axis  
 
    Returns
    -------
    cs : the coordinate system object
        The coordinate system object.
    """
    return topo.CoordinateSystem(origin, x_dir, y_dir)

def topo_attributes(topo, attributes):
    """
    This function add attributes to the list of topology.
 
    Parameters
    ----------
    topo : Topo Object
        Topo objects include Vertex, Edge, Wire, Face, Shell, Solid and Composite Topology Object.
        
    attributes : dictionary
        Dictionary of the attributes
 
    Returns
    -------
    vertex : vertex topology
        A vertex topology containing a point geometry
    """
    pass
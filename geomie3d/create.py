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
from . import topobj
from . import utility

def box(dimx, dimy, dimz, attributes = {}):
    """
    Constructs a box which is a solid topology where its bottom face midpt is at the origin (0,0,0).
 
    Parameters
    ----------
    dimx : float
        length of box in the x-axis.
    
    dimy : float
        length of box in the y-axis.
        
    dimz : float
        height of box.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    box : solid topology
        A box of solid topology
    """
    dimx_half = dimx/2
    dimy_half = dimy/2
    
    mnx = 0 - dimx_half
    mny = 0 - dimy_half
    mnz = 0
    
    mxx = 0 + dimx_half
    mxy = 0 + dimy_half
    mxz = dimz
    
    #bottom face
    xyz_list1 = np.array([[mnx, mny, mnz],
                          [mnx, mxy, mnz],
                          [mxx, mxy, mnz],
                          [mxx, mny, mnz]])
    
    vlist1 = vertex_list(xyz_list1)
    face1 = polygon_face_frm_verts(vlist1)
    
    #left vertical face
    xyz_list2 = np.array([[mnx, mxy, mnz],
                          [mnx, mny, mnz],
                          [mnx, mny, mxz],
                          [mnx, mxy, mxz]])
    
    vlist2 = vertex_list(xyz_list2)
    face2 = polygon_face_frm_verts(vlist2)
    
    #south vertical face
    xyz_list3 = np.array([[mnx, mny, mnz],
                          [mxx, mny, mnz],
                          [mxx, mny, mxz],
                          [mnx, mny, mxz]])
    
    vlist3 = vertex_list(xyz_list3)
    face3 = polygon_face_frm_verts(vlist3)
    
    #right vertical face
    xyz_list4 = np.array([[mxx, mny, mnz],
                          [mxx, mxy, mnz],
                          [mxx, mxy, mxz],
                          [mxx, mny, mxz]])
    
    vlist4 = vertex_list(xyz_list4)
    face4 = polygon_face_frm_verts(vlist4)
    
    #north vertical face
    xyz_list5 = np.array([[mxx, mxy, mnz],
                          [mnx, mxy, mnz],
                          [mnx, mxy, mxz],
                          [mxx, mxy, mxz]])
    
    vlist5 = vertex_list(xyz_list5)
    face5 = polygon_face_frm_verts(vlist5)
    
    #top cap face
    xyz_list6 = np.array([[mnx, mny, mxz],
                          [mxx, mny, mxz],
                          [mxx, mxy, mxz],
                          [mnx, mxy, mxz]])
    
    vlist6 = vertex_list(xyz_list6)
    face6 = polygon_face_frm_verts(vlist6)
    shell = topobj.Shell(np.array([face1, face2, face3, face4, face5, face6]))
    solid = topobj.Solid(shell)
    return solid
    
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
    vertex = topobj.Vertex(point, attributes = attributes)
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

def polygon_face_frm_verts(vertex_list, hole_vertex_list = [], attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology. 
        
    hole_vertex_list : a 2d list of vertices
        A 2d list of vertex topologies that is the hole of the face
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    bdry_wire = wire_frm_vertices(vertex_list)
    
    hole_wire_list = []
    for hole in hole_vertex_list:
        hole_wire =  wire_frm_vertices(vertex_list)
        hole_wire_list.append(hole_wire)
    
    face = polygon_face_frm_wires(bdry_wire, hole_wire_list = hole_wire_list, attributes = attributes)
    return face

def polygon_face_frm_wires(bdry_wire, hole_wire_list = [], attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    bdry_wire : a wire
        A wire topology that is the boundary of the face
        
    hole_wire_list : a list of wires
        A list of wire topologies that is the hole of the face
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    face = topobj.Face(attributes = attributes)
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
    edge = topobj.Edge(attributes = attributes)
    edge.add_polyline_curve(vertex_list)
    return edge

def wire_frm_vertices(vertex_list, attributes = {}):
    """
    Constructs a wire from a list of vertices. Wire created from this will have edges containing lines (polylines with only 2 vertices, a straight line).
 
    Parameters
    ----------
    vertex_list : a list of vertices
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    wire : wire topology
        A wire topology 
    """
    edge_list = []
    n_v = len(vertex_list)
    for cnt,v in enumerate(vertex_list):
        if cnt != n_v-1:
            edge = pline_edge_frm_verts([v, vertex_list[cnt+1]])
        else:
            edge = pline_edge_frm_verts([v, vertex_list[0]])
        edge_list.append(edge)
    
    wire = wire_frm_edges(edge_list, attributes = attributes)
    
    return wire

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
    wire = topobj.Wire(edge_list, attributes = attributes)
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
    return utility.CoordinateSystem(origin, x_dir, y_dir)

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

def composite(topo_list, attributes={}):
    """
    This function add attributes to the list of topology.
 
    Parameters
    ----------
    topo_list : Topo Object List
        List of Topo objects include Vertex, Edge, Wire, Face, Shell, Solid and Composite Topology Object.
        
    attributes : dictionary
        Dictionary of the attributes
 
    Returns
    -------
    composite : composite topology
        A composite topology containing the topo list
    """
    return topobj.Composite(topo_list, attributes = attributes)
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
from . import geom
from . import get
import numpy as np

class Topology(object):
    """
    base class of all topology
    
    Parameters
    ----------
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
        
    Attributes
    ----------    
    attributes : dictionary
        The dictionary of attributes appended to the object.
    
    is_topo : bool
        an attribute to identify the object as a topology object.
        
    """
    def __init__(self, attributes = {}):
        """Initialises the class"""
        self.attributes = attributes
        self.is_topo = True
    
    def update_attributes(self, new_attributes):
        """
        This function updates the attributes.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes

class Vertex(Topology):
    """
    A vertex object
    
    Parameters
    ----------
    point : Point Object
        The Point Geometry
        
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------    
    point : Point Object
        The Point Object.
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, point, attributes = {}):
        """Initialises the class"""
        super(Vertex, self).__init__(attributes = attributes)
        self.point = point
        
class Edge(Topology):
    """
    An edge object
    
    Parameters
    ----------
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------
    curve_type : interger
        0 = polyline
        
    curve : Curve Object
        The curve Geometry
        
    vertex_list : list of vertex Object
        The vertices that defines the curve in the edge.
    
    start_vertex : vertex object
        The starting vertex.
    
    end_vertex : vertex object
        The end vertex.
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, attributes = {}):
        """Initialises the class"""
        super(Edge, self).__init__(attributes = attributes)
        self.curve_type = None
        self.curve = None
        self.vertex_list = None
        self.start_vertex = None
        self.end_vertex = None
        
    def add_polyline_curve(self, vertex_list):
        """
        This function creates a polyline edge topology.
     
        Parameters
        ----------
        vertex_list : list of vertex Object
            The vertices that defines the polyline in the edge.
        """
        self.curve_type = 0
        self.vertex_list = vertex_list
        self.start_vertex = vertex_list[0]
        self.end_vertex = vertex_list[-1]
        point_list = [v.point for v in vertex_list]
        curve = geom.PolylineCurve(point_list)
        self.curve = curve
        
class Wire(Topology):
    """
    A wire object
    
    Parameters
    ----------
    edge_list : List of Edge Objects
        List of Edge Topology
    
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------    
    edge_list : List of Edge Objects
        List of Edge Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, edge_list, attributes = {}):
        """Initialises the class"""
        super(Wire, self).__init__(attributes = attributes)
        self.edge_list = edge_list
        
class Face(Topology):
    """
    A face object
    
    Parameters
    ----------
    surface : Surface Object
        The Surface Geometry
        
    attributes : dictionary, optional
        The dictionary of attributes appended to the object.
    
    Attributes
    ----------
    surface_type : interger
        0 = polygon
        
    surface : Surface Object
        The Surface Geometry
        
    bdry_wire : Wire Object
        The wire object that defines the outer boundary of the face.
    
    hole_wire_list : list of wire Object
        The list of wire objects that define the holes in the face.
    
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, attributes = {}):
        """Initialises the class"""
        super(Face, self).__init__(attributes = attributes)
        self.surface_type = None
        self.surface = None
        self.bdry_wire = None
        self.hole_wire_list = None
        
    def add_polygon_surface(self, bdry_wire, hole_wire_list = []):
        """
        This function adds a polygon to the face object.
     
        Parameters
        ----------
        bdry_wire : Wire Object
        The wire object that defines the outer boundary of the face.
    
        hole_wire_list : list of wire Object, optional
            The list of wire objects that define the holes in the face.
     
        """
        self.surface_type = 0
        self.bdry_wire = bdry_wire
        self.hole_wire_list = hole_wire_list
        #get the point list from the bdry wire
        pt_list = get.get_points_frm_wire(bdry_wire)
        
        hole_point_2dlist = [get.get_points_frm_wire(hole_wire) 
                             for hole_wire in hole_wire_list]
       
        surface = geom.PolygonSurface(pt_list, 
                                      hole_point_2dlist = hole_point_2dlist)
        self.surface = surface
        
class Shell(Topology):
    """
    A shell object
    
    Parameters
    ----------
    face_list : List of Face Objects
        List of Face Topology
    
    Attributes
    ----------    
    face_list : List of Face Objects
        List of Face Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, face_list, attributes = {}):
        """Initialises the class"""
        super(Shell, self).__init__(attributes = attributes)
        self.face_list = face_list
        self.attributes = attributes

class Solid(Topology):
    """
    A solid object
    
    Parameters
    ----------
    shell : Shell Object
        Shell Topology
    
    Attributes
    ----------    
    shell : Shell Object
        Shell Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, shell, attributes = {}):
        """Initialises the class"""
        super(Solid, self).__init__(attributes = attributes)
        self.shell = shell
        self.attributes = attributes

class Composite(Topology):
    """
    A composite object
    
    Parameters
    ----------
    topology_list : List of Topology
        List of Topology
    
    Attributes
    ----------    
    topology_list : List of Topology
        List of Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, topology_list, attributes = {}):
        """Initialises the class"""
        super(Composite, self).__init__(attributes = attributes)
        self.topology_list = topology_list   
        
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
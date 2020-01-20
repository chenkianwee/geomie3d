# ==================================================================================================
#
#    Copyright (c) 2020, Chen Kian Wee (chenkianwee@gmail.com)
#
#    This file is part of geomy3d
#
#    geomy3d is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    geomy3d is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with py4design.  If not, see <http://www.gnu.org/licenses/>.
#
# ==================================================================================================
class Vertex(object):
    """
    A vertex object
    
    Parameters
    ----------
    point : Point Object
        The Point Geometry
    
    Attributes
    ----------    
    point : Point Object
        The Point Object.
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, point, attributes = {}):
        """Initialises the class"""
        self.point = point
        self.attributes = attributes
        
class Edge(object):
    """
    An edge object
    
    Parameters
    ----------
    curve : Curve Object
        The Curve Geometry
    
    Attributes
    ----------    
    curve : Curve Object
        The Curve Geometry.
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, curve, attributes = {}):
        """Initialises the class"""
        self.curve = curve
        self.attributes = attributes
        
class Wire(object):
    """
    A wire object
    
    Parameters
    ----------
    edge_list : List of Edge Objects
        List of Edge Topology
    
    Attributes
    ----------    
    edge_list : List of Edge Objects
        List of Edge Topology
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, edge_list, attributes = {}):
        """Initialises the class"""
        self.edge_list = edge_list
        self.attributes = attributes
        
class Face(object):
    """
    A face object
    
    Parameters
    ----------
    surface : Surface Object
        The Surface Geometry
    
    Attributes
    ----------    
    surface : Surface Object
        The Surface Geometry
        
    attributes : dictionary
        The dictionary of attributes appended to the object.
    """
    def __init__(self, surface, attributes = {}):
        """Initialises the class"""
        self.surface = surface
        self.attributes = attributes
        
class Shell(object):
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
        self.face_list = face_list
        self.attributes = attributes

class Solid(object):
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
        self.shell = shell
        self.attributes = attributes

class Composite(object):
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
        self.topology_list = topology_list
        self.attributes = attributes      
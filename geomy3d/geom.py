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

class Point(object):
    """
    A point geometry
    
    Parameters
    ----------
    xyz : tuple
        The xyz coordinates
    
    Attributes
    ----------    
    xyz : tuple
        The xyz coordinates
        
    """
    def __init__(self, xyz):
        """Initialises the class"""
        self.xyz = xyz
        
class PolylineCurve(object):
    """
    A curve geometry
    
    Parameters
    ----------
    point_list : List of Point Geometry
        List of Point Geometry
        
    Attributes
    ----------    
    point_list : List of Point Geometry
        List of Point Geometry
    
    """
    def __init__(self, point_list):
        """Initialises the class"""
        self.point_list = point_list
        
class PolygonSurface(object):
    """
    A surface geometry
    
    Parameters
    ----------
    point_list : List of Point Geometry
        List of Point Geometry
    
    Attributes
    ----------    
    point_list : List of Point Geometry
        List of Point Geometry
        
    """
    def __init__(self, point_list):
        """Initialises the class"""
        self.point_list = point_list
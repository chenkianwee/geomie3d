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
from enum import Enum
from . import calculate
import numpy as np 

class Point(object):
    def __init__(self, xyz: np.ndarray):
        """
        A point geometry
        
        Parameters
        ----------
        xyz : tuple
            The xyz coordinates
        """
        if type(xyz) != np.ndarray:
            xyz = np.array(xyz)

        self.xyz: np.ndarray = xyz
        """array specifying the xyz coordinates"""
        
class CurveType(Enum):
    """
    There are two types of curve types
    """
    POLYLINE = 0
    """POLYLINE = 0"""
    BSPLINE = 1
    """BSPLINE = 1"""

class PolylineCurve(object):
    def __init__(self, point_list: list[Point]):
        """
        A curve geometry
        
        Parameters
        ----------
        point_list :list[Point]
            list[Point]
        """
        if type(point_list) != np.ndarray:
            point_list = np.array(point_list)

        self.curve_type: CurveType = CurveType.POLYLINE
        """Curve type POLYLINE"""
        self.point_list: list[Point] = point_list
        """list[Point]"""
        
class Surface(object):
    def __init__(self):
        """
        base class of all surface geometry
        """
        self.normal: np.ndarray = None
        """normal of the surface"""


class SrfType(Enum):
    """
    There are two types of surfaces
    """
    POLYGON = 0
    """POLYGON = 0"""
    BSPLINE = 1
    """BPSLINE = 1"""

class PolygonSurface(Surface):
    def __init__(self, point_list: list[Point], hole_point_2dlist: list[list[Point]] = []):
        """
        A polygon surface geometry
        
        Parameters
        ----------
        point_list : list[Point]
            List of Point Geometry defining the boundary of the polygon
            
        hole_point_2dlist : list[list[Point]], optional
            2d list of Point Geometry defining the holes of the polygon

        """
        super(PolygonSurface, self).__init__()
        if type(point_list) != np.ndarray:
            point_list = np.array(point_list)
        
        self.point_list: list[Point] = point_list
        """List of Point Geometry defining the boundary of the polygon"""

        self.surface_type: SrfType = SrfType.POLYGON
        """Surface type POLYGON"""

        self.hole_point_2dlist: list[list[Point]] = None
        """2d list of Point Geometry defining the holes of the polygon"""

        if len(hole_point_2dlist) !=0:
            if type(hole_point_2dlist) != np.ndarray:
                hole_point_2dlist = np.array(hole_point_2dlist)
            self.hole_point_2dlist = hole_point_2dlist
        
        self.calc_normal()
    
    def calc_normal(self):
        """
        calculates the normal of the surface
        """
        point_list = self.point_list
        #calculate the normal of the surface
        xyz_list = [point.xyz for point in point_list]
        vector1 = xyz_list[1] - xyz_list[0]
        vector2 = xyz_list[-1] - xyz_list[0]
        normal = calculate.cross_product(vector1, vector2)
        normal = calculate.normalise_vectors(normal)
        self.normal = normal
    
    def update(self, point_list: list[Point],  hole_point_2dlist: list[list[Point]] = []):
        """
        updates the boundary and holes of the surface
        
        Parameters
        ----------
        point_list : list[Point]
            List of Point Geometry defining the boundary of the polygon
            
        hole_point_2dlist : list[list[Point]], optional
            2d list of Point Geometry defining the holes of the polygon
        """
        self.point_list = point_list
        self.hole_point_2dlist = None
        if len(hole_point_2dlist) !=0:
            if type(hole_point_2dlist) != np.ndarray:
                hole_point_2dlist = np.array(hole_point_2dlist)
            self.hole_point_2dlist = hole_point_2dlist
            
        self.calc_normal()
        
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
"""
Py3dmodel
================================================
Documentation is available in the docstrings and online at https://xxx.xxx.xx

Submodules
-----------
::
    
 construct                    --- Functions for constructing geometries. e.g. construct a face
                                  dependencies: scipy, numpy (for delaunay function only)
 fetch                        --- Functions for fetching information from the geometries. e.g. fetch points from face
 modify                       --- Functions for modifying the geometries. e.g. move the face from pointA to pointB
 calculate                    --- Functions for obtaining information from geometries through calculations.
                                  e.g. calculate the mid point of a face
 utility                      --- Functions that does not fit into the previous four modules. e.g. visualise the face, export the face to stl
 export_collada               --- Functions to export the geometries to collada format.
                                  dependencies: pycollada
"""

from . import get
from . import calculate
from . import modify
from . import create
from . import utility

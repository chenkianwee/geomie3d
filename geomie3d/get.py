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
from . import modify

def get_points_frm_wire(wire):
    edge_list = wire.edge_list
    #TODO currently only account for polyline edges
    # vertex_2dlist = np.array([edge.vertex_list if edge.curve_type == 0 
    #                           else edge.vertex_list for edge in edge_list 
    #                           ])
    
    vertex_2dlist = np.array([edge.vertex_list for edge in edge_list 
                              if edge.curve_type == 0])
    
    vertex_list = vertex_2dlist.flatten()
    point_list = np.array([v.point for v in vertex_list])
    #fused the points
    fused_vertices = modify.fuse_points(point_list)
    return fused_vertices

def get_vertices_frm_wire(wire):
    edge_list = wire.edge_list
    #TODO currently only account for polyline edges
    # vertex_2dlist = np.array([edge.vertex_list if edge.curve_type == 0 
    #                           else edge.vertex_list for edge in edge_list 
    #                           ])
    
    vertex_2dlist = np.array([edge.vertex_list for edge in edge_list 
                              if edge.curve_type == 0])
    vertex_list = vertex_2dlist.flatten()
    #fused the points
    fused_vertices = modify.fuse_vertices(vertex_list)
    return fused_vertices

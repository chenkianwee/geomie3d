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
from scipy.spatial import Delaunay

from . import get
from . import create
from . import utility

def fuse_vertices(vertex_list):
    """
    This function fuses the vertices, duplicate vertices are fuse into a single vertex.
    
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology to be fused.
 
    Returns
    -------
    fused_vertex_list : ndarray of vertex topology
        A numpy array of the fused vertices
    """
    xyz_list = np.array([v.point.xyz for v in vertex_list])
    vals, u_ids, inverse = np.unique(xyz_list, axis = 0,
                                      return_inverse=True,
                                      return_index=True)
    
    dup_ids = utility.id_dup_indices_1dlist(inverse)
    dups = vertex_list[dup_ids]
    cnt = 0
    for d in dups:
        d_id = dup_ids[cnt]
        att_list = [v.attributes for v in d]
        new_dict = {"fused_vertex"+str(x):att_list[x] 
                    for x in range(len(att_list)) if att_list[x]}
        vertex_list[d_id[0]].update_attributes(new_dict)
        cnt+=1
        
    u_ids = np.sort(u_ids)
    unique_v = vertex_list[u_ids]
    return unique_v

def fuse_points(point_list):
    """
    This function fuses the points, duplicate points are fuse into a single point.
    
    Parameters
    ----------
    point_list : a list of point
        A list of point object to be fused.
 
    Returns
    -------
    fused_point_list : ndarray of point object
        A numpy array of the fused points
    """
    
    xyz_list = np.array([point.xyz for point in point_list])
    vals, u_ids, inverse = np.unique(xyz_list, axis = 0,
                                      return_inverse=True,
                                      return_index=True)        
    u_ids = np.sort(u_ids)
    unique_p = point_list[u_ids]
    return unique_p

def delaunay_face(face, tolerance = 1e-6):
    """
    This function triangulate a face.
 
    Parameters
    ----------
    face : face object
        the face object to be triangulate.
        
    tolerance : float, optional
        The minimal surface area of each triangulated face, Default = 1e-06.. Any faces smaller than the tolerance will be deleted.
 
    Returns
    -------
    list of face : list of face
        A list of triangles constructed from the meshing.
    """
    #TODO:need to think about the inheritance of the vertices, edges and wires 
    srf_type = face.surface_type
    if srf_type == 0:
        bdry_wire = face.bdry_wire
        bdry_verts = get.get_vertices_frm_wire(bdry_wire)
        xyz_list = np.array([v.point.xyz for v in bdry_verts])
        
        hole_wire_list = face.hole_wire_list
        hole_verts = []
        hole_xyz_list = np.array([pt.xyz for hole in hole_list for 
                                  pt in hole])
        
        #TODO transform the points from 3d to 2d
        xy_list = np.delete(xyz_list, 2, axis=1)
        hole_xy_list = np.delete(hole_xyz_list, 2, axis=1)
        
        d_xy_list = np.concatenate((xy_list, hole_xy_list))
        tri = Delaunay(d_xy_list)
        
        chosen = d_xy_list[tri.simplices]
        # print(chosen)
        # for indices in tri.simplices:
        #     fp = d_xy_list[indices]
        #     print(fp)
            # create.polygon_face_frm_verts(vertex_list)
            # pt1 = list(xyz[verts[0]])
            # pt2 = list(xyz[verts[1]])
            # pt3 = list(xyz[verts[2]])
            # occtriangle = make_polygon([pt1,pt2,pt3])
            # tri_area = calculate.face_area(occtriangle)
            # if tri_area > tolerance:
            #     occtriangles.append(occtriangle)
            
def face2mesh(face):
    """
    This function converts a face to a mesh for visualisation.
 
    Parameters
    ----------
    face : face object
        the face object to be triangulated.The face must be single facing, it
        cannot have regions that are facing itself.
        
    tolerance : float, optional
        The minimal surface area of each triangulated face, Default = 1e-06.. Any faces smaller than the tolerance will be deleted.
 
    Returns
    -------
    mesh_dictionary : dictionary
        vertices: ndarray of shape(Npoints,3) of the vertiices
        indices: ndarray of shape(Ntriangles,3) indices of the triangles.
    """
    srf_type = face.surface_type
    if srf_type == 0:
        #TODO TRANSFORM THE FACE FROM 3D TO 2D FOR THE DELAUNAY
        
        #get the boundary wire
        bdry_wire = face.bdry_wire
        bdry_verts = get.get_vertices_frm_wire(bdry_wire)
        xyz_list = np.array([v.point.xyz for v in bdry_verts])
        nxyz = len(xyz_list)
        #get all the hole wires
        hole_wire_list = face.hole_wire_list
        holev_2darr = np.array([get.get_vertices_frm_wire(w) for w 
                               in hole_wire_list])
        #get the indices of the hole wires in 2d
        n_hole_wires = len(holev_2darr)
        holei_2darr = np.array([range(nxyz, len(holev_2darr[i]) + nxyz) if i == 0 
                                else range(len(holev_2darr[i-1]), len(holev_2darr[i]) + nxyz + len(holev_2darr[i-1]))
                                for i in range(n_hole_wires)])
        #flatten the hole wire verts for performing delaunay
        hole_verts = holev_2darr.flatten()
        hole_xyz_list = np.array([v.point.xyz for v in hole_verts])
        
        #add all the verts together
        mesh_xyz_list = np.concatenate((xyz_list, hole_xyz_list))
        nmesh_xyz = len(mesh_xyz_list)
        
        #perform delaunay 
        xy_list = np.delete(xyz_list, 2, axis=1)
        hole_xy_list = np.delete(hole_xyz_list, 2, axis=1)
        d_xy_list = np.concatenate((xy_list, hole_xy_list))
        tri = Delaunay(d_xy_list)
        indices2d = tri.simplices
        tri_indices = np.array(range(0,len(indices2d)))
        #remove all the faces that are holes
        tri_inds_remove = np.array([], dtype = "int64")
        for h in holei_2darr:
            isin = np.isin(indices2d, h)
            isall = np.all(isin, axis=1)
            tri_ind_remove = np.where(isall)
            tri_inds_remove = np.append(tri_inds_remove, tri_ind_remove)
    
        tri_inds_remove = np.unique(tri_inds_remove)
        indices2d = np.delete(indices2d, tri_inds_remove, axis=0)
        return {"vertices":mesh_xyz_list, "indices":indices2d}
    else:
        return None
    
def combine_meshes(mesh_dict_list):
    pass

def trsf_cs(cs1, cs2, topo):
    pass
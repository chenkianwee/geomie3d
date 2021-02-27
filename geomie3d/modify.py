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
from scipy import spatial

from . import get
from . import create
from . import utility
from . import topobj
from . import geom
from . import earcut
from . import calculate

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

def triangulate_face(face, indices = False):
    """
    Triangulates a face.
 
    Parameters
    ----------
    face : topo object
        the face object to be triangulated.
        
    indices : bool, optional
        Specify whether to an array of [xyzs, indices]. If True will not return the points. Default = False.
 
    Returns
    -------
    list of face : list of face
        A list of triangles constructed from the meshing.
    """
    srf_type = face.surface_type
    if srf_type == geom.SrfType.POLYGON:
        bdry_verts = get.bdry_vertices_frm_face(face)
        nbdry_verts = len(bdry_verts)
        
        hole_verts_2dlist = get.hole_vertices_frm_face(face)
        nholes = len(hole_verts_2dlist)
        if nholes > 0:
            hole_verts_list = hole_verts_2dlist.flatten()
            all_verts = np.append(bdry_verts, hole_verts_list)
            
            nhole_verts_ls = [len(hole_verts) for hole_verts in hole_verts_2dlist]
            hole_idxs = [nbdry_verts]
            for cnt,nhole_verts in enumerate(nhole_verts_ls):
                if cnt != nholes-1:    
                    hole_idx = nbdry_verts + nhole_verts
                    hole_idxs.append(hole_idx)
            
        else:
            all_verts = bdry_verts
            hole_idxs = None
        
        all_xyzs_2dlist = np.array([vert.point.xyz for vert in all_verts])
        
        #if face is not facing z rotate the xyzs here 
        nrml = get.face_normal(face)
        angle = calculate.angle_btw_2vectors([0,0,1], nrml)
        if 180 > angle > 0:
            #means it is not facing z up
            #turn it to z-up
            axis = calculate.cross_product([0,0,1], nrml)
            rot_matrice = calculate.rotate_matrice(axis, angle)
            xyz2tri_2darr = calculate.trsf_xyzs(all_xyzs_2dlist, rot_matrice)
        else:
            xyz2tri_2darr = all_xyzs_2dlist
        
        #FLATTEN ALL THE XYZS AND THEN TRIANGULATE THEM WITH EARCUT
        all_xyzs = xyz2tri_2darr.flatten().tolist()
        tri_idxs = earcut.earcut(all_xyzs, hole_idxs, 3)
        ntri_idxs = len(tri_idxs) 
        if ntri_idxs > 0:
            #if it is successfully triangulated
            shape1 = int(ntri_idxs/3)
            
            #check if the vertices is ccw
            tri_idxs_arr = np.array(tri_idxs)
            tri_xyzs = np.take(all_xyzs_2dlist, tri_idxs_arr, axis=0)
            tri_xyzs2 = np.reshape(tri_xyzs,(shape1,3,3))
            tri1 = tri_xyzs2[0]
            is_ccw = calculate.is_anticlockwise(tri1, nrml)
            
            #if it is not ccw flip it to make it ccw
            tri_idxs_2darr = np.reshape(tri_idxs_arr, (shape1,3))
            if is_ccw == False:
                tri_idxs_2darr = np.flip(tri_idxs_2darr, axis = 1)
                
            if indices == False:
                #TODO figure out the inheritance structure
                #reconstruct the triangulated faces 
                tri_idxs_arr = tri_idxs_2darr.flatten()
                tri_verts_arr = np.take(all_verts, tri_idxs_arr, axis = 0)
                tri_verts_2darr = np.reshape(tri_verts_arr,(shape1,3))
                face_atts = face.attributes
                tri_faces = []
                for tri_verts in tri_verts_2darr:    
                    tri_face = create.polygon_face_frm_verts(tri_verts, attributes = face_atts)
                    tri_faces.append(tri_face)
                return tri_faces
            else:
                return [all_xyzs_2dlist, tri_idxs_2darr]
        else:
            return []

def shell_frm_delaunay(vertex_list, tolerance = 1e-6):
    """
    This function creates a TIN from a vertex list.
 
    Parameters
    ----------
    vertex_list : face object
        the x and y dim of the vertex has to be on the same plane.
 
    Returns
    -------
    shell : shell topology
        A shell object.
    """
    #TODO:need to think about the inheritance of the vertices, edges and wires 
    srf_type = face.surface_type
    if srf_type == geom.SrfType.POLYGON:
        nrml = face.surface.normal
        if nrml != np.array([0,0,1]) or nrml != np.array([0,0,-1]):
            #it must be transformed to be flat
            pass
        
        bdry_wire = face.bdry_wire
        bdry_verts = get.vertices_frm_wire(bdry_wire)
        xyz_list = np.array([v.point.xyz for v in bdry_verts])
        
        hole_wire_list = face.hole_wire_list
        
        hole_verts = []
        hole_xyz_list = np.array([pt.xyz for hole in hole_list for 
                                  pt in hole])
        
        
        
        #TODO transform the points from 3d to 2d
        xy_list = np.delete(xyz_list, 2, axis=1)
        hole_xy_list = np.delete(hole_xyz_list, 2, axis=1)
        
        d_xy_list = np.concatenate((xy_list, hole_xy_list))
        tri = spatial.Delaunay(d_xy_list)
        
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
            
def faces2mesh(face_list):
    """
    This function converts faces to a mesh for visualisation.
 
    Parameters
    ----------
    face_list : list of face object
        the list of face object to be triangulated.The face must be single facing, it
        cannot have regions that are facing itself.
 
    Returns
    -------
    mesh_dictionary : dictionary
        vertices: ndarray of shape(Npoints,3) of the vertiices
        indices: ndarray of shape(Ntriangles,3) indices of the triangles.
    """
    all_xyzs = []
    all_idxs = []
    idx_cnt = 0
    for f in face_list:
        srf_type = f.surface_type
        if srf_type == geom.SrfType.POLYGON:
            xyzs_indxs = triangulate_face(f, indices = True)
            if len(all_xyzs) == 0:
                all_xyzs = xyzs_indxs[0]
            else:
                all_xyzs = np.append(all_xyzs, xyzs_indxs[0], axis=0)
            
            if len(all_idxs) == 0:
                all_idxs = xyzs_indxs[1]
            else:
                indxs_new =  xyzs_indxs[1] + idx_cnt
                all_idxs = np.append(all_idxs, indxs_new, axis=0)
            
            idx_cnt += len(xyzs_indxs[0])
    
    return {"vertices":all_xyzs, "indices":all_idxs}

def edges2lines(edge_list):
    """
    converts edges to lines for visualisation.
 
    Parameters
    ----------
    edge_list : list of edge object
        the list of edge object to convert to lines.
 
    Returns
    -------
    vertices : shape(Npoints,3) of the vertiices
        vertices of the lines
    """
    all_xyzs = []
    
    for e in edge_list:
        vs = get.vertices_frm_edge(e)
        xyzs = np.array([v.point.xyzs for v in vs])
        #repeat each point and remove the first and last point after the repeat
        xyzs = np.repeat(xyzs, 2, axis=0)[1:-1]
        if len(all_xyzs) == 0:
            all_xyzs = xyzs
        else:
            all_xyzs = np.append(all_xyzs, xyzs)
    
    return all_xyzs 

def trsf_cs(cs1, cs2, topo):
    pass
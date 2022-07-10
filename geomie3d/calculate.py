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
import math
from itertools import chain
import numpy as np

from . import topobj
from . import get
from . import utility
from . import modify
from . import geom

def dist_btw_xyzs(xyzs1, xyzs2):
    """
    This function calculates the distance between two xyz.
 
    Parameters
    ----------    
    xyzs1 : ndarray
        array defining the point.
        
    xyzs2 : ndarray
        array defining the point.
 
    Returns
    -------
    distance : float
        the distance(s) between the points
    """
    if type(xyzs1) != np.ndarray:
        xyzs1 = np.array(xyzs1)
    if type(xyzs2) != np.ndarray:
        xyzs2 = np.array(xyzs2)
    
    xyzs12 = (xyzs2-xyzs1)**2
    nshape = len(xyzs12.shape)
    if nshape > 1:    
        xyzsum = xyzs12.sum(axis=1)
    else:
        xyzsum = xyzs12.sum()
    # print(xyzsum)
    distance = np.sqrt(xyzsum)
    return distance

def xyzs_mean(xyzs):
    """
    This function calculates the mean of all points. 
 
    Parameters
    ----------
    xyzs : ndarray
        array defining the point.
    
    Returns
    -------
    midpt : xyz
        The mean of all the points.
    """

    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
        
    zip_xyzs = xyzs.T
    
    x_list = zip_xyzs[0]
    y_list = zip_xyzs[1]
    z_list = zip_xyzs[2]
        
    npts = len(x_list)
    x_mean = (np.sum(x_list))/npts
    y_mean = (np.sum(y_list))/npts
    z_mean = (np.sum(z_list))/npts
    
    centre_pt = np.array([x_mean, y_mean, z_mean])

    return centre_pt

def bbox_frm_xyzs(xyzs):
    """
    This function returns the bbox of the xyzs.
    
    Parameters
    ----------
    xyzs : ndarray
        array defining the point.

    Returns
    -------
    bbox : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
    """
    import numpy as np
    
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    xyzs_t = xyzs.T
    x = xyzs_t[0]
    y = xyzs_t[1]
    z = xyzs_t[2]
    
    mnx = np.amin(x)
    mxx = np.amax(x)
    
    mny = np.amin(y)
    mxy = np.amax(y)
    
    mnz = np.amin(z)
    mxz = np.amax(z)
    
    return np.array([mnx,mny,mnz,mxx,mxy,mxz])
    
def bbox_centre(bbox):
    """
    This function returns the centre point of the bbox .
    
    Parameters
    ----------
    bbox : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.

    Returns
    -------
    xyz : ndarray
        array defining the point.
    """
    midx = bbox[0] + ((bbox[3] - bbox[0])/2)
    midy = bbox[1] + ((bbox[4] - bbox[1])/2)
    midz = bbox[2] + ((bbox[5] - bbox[2])/2)
    
    return np.array([midx,midy,midz])
    
def bbox_frm_bboxes(bbox_list):
    """
    This function recalculate the bbox from a list of bboxes.
    
    Parameters
    ----------
    bbox_list : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.

    Returns
    -------
    bbox : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
    """
    zip_box = list(zip(*bbox_list))
    mnx = min(zip_box[0])
    mny = min(zip_box[1])
    mnz = min(zip_box[2])
    
    mxx = max(zip_box[3])
    mxy = max(zip_box[4])
    mxz = max(zip_box[5])
    
    return np.array([mnx,mny,mnz,mxx,mxy,mxz])

def is_collinear(vertex_list):
    """
    This function checks if the list of points are collinear. 
 
    Parameters
    ----------
    vertex_list : ndarray
        array of vertices.
        
    Returns
    -------
    True or False : bool
        If True the list of points are collinear.
    """
    from sympy import Point3D
    sympts = [Point3D(v.point.xyz) for v in vertex_list]
    collinear = Point3D.are_collinear(*sympts)
    return collinear

def is_coplanar(vertex_list):
    """
    This function checks if the list of points are coplanar. 
 
    Parameters
    ----------
    vertex_list : ndarray
        array of vertices.
        
    Returns
    -------
    True or False : bool
        If True the list of points are coplanar.
    """
    from sympy import Point3D
    sympts = [Point3D(v.point.xyz) for v in vertex_list]
    
    coplanar = Point3D.are_coplanar(*sympts)
    return coplanar

def bbox_frm_topo(topo):
    """
    calculate the bbox from a topology object.
    
    Parameters
    ----------
    topo : topology object
        The topology to analyse

    Returns
    -------
    bbox : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
    """
    verts = get.topo_explorer(topo, topobj.TopoType.VERTEX)
    xyzs = np.array([v.point.xyz for v in verts])
    bbox = bbox_frm_xyzs(xyzs)
    return bbox

def xyzs_in_bbox(xyzs, bbox, zdim = True, indices = False):
    """
    This function if the points are within the given boundary box.
 
    Parameters
    ----------
    xyzs : ndarray
        array defining the points.
    
    bbox : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
        
    zdim : bool, optional
        If True will check the z-dimension.
        
    indices : bool, optional
        Specify whether to return the indices of the points in the boundary. If True will not return the points. Default = False.
    
    Returns
    -------
    points_in_bdry : pyptlist or indices
        The points that is in the boundary. If indices==True, this will be the indices instead of the actual points.
    """
    import numpy as np
    
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    zip_xyzs = xyzs.T
    xlist = zip_xyzs[0] 
    ylist = zip_xyzs[0]
    zlist = zip_xyzs[0]
    
    mnx = bbox[0]
    mny = bbox[1]
    mxx = bbox[3]
    mxy = bbox[4]
    
    x_valid = np.logical_and((mnx <= xlist),
                             (mxx >= xlist))
    
    y_valid = np.logical_and((mny <= ylist),
                             (mxy >= ylist))
    
    if zdim == True:
        mnz = bbox[2]
        mxz = bbox[5]
        
        z_valid = np.logical_and((mnz <= zlist),
                                 (mxz >= zlist))
    
        index_list = np.where(np.logical_and(x_valid, y_valid, z_valid))[0]
    
    else:
        index_list = np.where(np.logical_and(x_valid, y_valid))[0]
    
    if indices == True:
        if index_list.size > 0:
            return index_list
        else:
            return []
    else:
        pts_in_bdry = np.take(xyzs, index_list, axis = 0)
        return pts_in_bdry

def is_xyz_in_bbox(xyz, bbox, zdim = True):
    """
    This function check if a point is in bounding box.  
 
    Parameters
    ----------
    xyz : ndarray
        array defining the point.
        
    bbox : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
        
    zdim : bool, optional
        If True will check the z-dimension.
    
    Returns
    -------
    in_boundary : bool
        True or False, is the point in bounding box.
    """
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    mnx = bbox[0]
    mny = bbox[1]
    mnz = bbox[2]
    mxx = bbox[3]
    mxy = bbox[4]
    mxz = bbox[5]
    
    in_bdry = False
    if zdim == True:
        if mnx<=x<=mxx and mny<=y<=mxy and mnz<=z<=mxz:
            in_bdry = True
        
    else:
        if mnx<=x<=mxx and mny<=y<=mxy:
            in_bdry = True
    
    return in_bdry
    
def match_xyzs_2_bboxes(xyzs, bbox_list, zdim = True):
    """
    This function returns the point indices follow by the bbox indices which it is contained in.
    
    Parameters
    ----------
    xyzs : ndarray
        array defining the points.
    
    bbox_list : tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
        
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    point_bbox_indices : nparray
        point indices follow by the bbox indices.
    """
    import numpy as np
    
    if type(bbox_list) != np.ndarray:
        bbox_list = np.array(bbox_list)
        
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
        
    def reshape_bdry(nparray, repeat):
        arrx = np.expand_dims(nparray, axis=0)
        arrx = np.repeat(arrx, repeat, axis=0)
        return arrx
    
    npts = len(xyzs)
    tbdry = bbox_list.T
    
    xmin_list = reshape_bdry(tbdry[0], npts)
    ymin_list = reshape_bdry(tbdry[1], npts)
    xmax_list = reshape_bdry(tbdry[3], npts)
    ymax_list = reshape_bdry(tbdry[4], npts)
    
    tpt = xyzs.T
    xlist = tpt[0]
    xlist.shape = (npts,1)
    
    ylist = tpt[1]
    ylist.shape = (npts,1)
    
    zlist = tpt[2]
    zlist.shape = (npts,1)
    
    x_valid = np.logical_and(xlist >= xmin_list,
                             xlist <= xmax_list)
    
    y_valid = np.logical_and(ylist >= ymin_list,
                             ylist <= ymax_list)
    
    if zdim == True:
        zmin_list = reshape_bdry(tbdry[2], npts)
        zmax_list = reshape_bdry(tbdry[5], npts)
        z_valid = np.logical_and(zlist >= zmin_list,
                                 zlist <= zmax_list)
        xyz_valid = np.logical_and(x_valid, y_valid, z_valid)
    else:
        xyz_valid = np.logical_and(x_valid, y_valid)
    
    index = np.where(xyz_valid)
    return index

def id_bboxes_contain_xyzs(bbox_list, xyzs, zdim = True):
    """
    This function returns the indices of the bbox which contains the points.
    
    Parameters
    ----------
    bbox_list : list of tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
        
    xyzs : ndarray
        array defining the points.
            
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    bbox_indices : nparray
        Indices of the boundary that contains the point.
    """
    import numpy as np
    indices = match_xyzs_2_bboxes(xyzs, bbox_list, zdim = zdim)
    bbox_indices = indices[1]
    bbox_indices = np.unique(bbox_indices)
    
    return bbox_indices

def id_xyzs_in_bboxes(xyzs, bbox_list, zdim = False):
    """
    This function returns the indices of the points that are contained by bboxes.
    
    Parameters
    ----------
    xyzs : ndarray
        array defining the points.
        
    bbox_list : list of tuple
        The tuple (xmin,ymin,zmin,xmax,ymax,zmax). The tuple specifies the boundaries.
        
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    pt_indices : nparray
        Indices of the points in the bboxes.
    """
    import numpy as np
    indices = match_xyzs_2_bboxes(xyzs, bbox_list, zdim = zdim)
    pt_indices = indices[0]
    pt_indices = np.unique(pt_indices)
    
    return pt_indices

def angle_btw_2vectors(vector1, vector2):
    """
    This function calculate the angle between two vectors.
 
    Parameters
    ----------    
    vector1 : ndarray
        array defining the vector(s).
        
    vector2 : ndarray
        array defining the vector(s).
 
    Returns
    -------
    angle : float
        angle between the two vectors
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
    
    nshape = len(vector1.shape)
    if nshape > 1:    
        dot = np.sum(vector1*vector2, axis=1)
        mag1 = np.sqrt(np.sum(vector1**2, axis=1))
        mag2 = np.sqrt(np.sum(vector2**2, axis=1))
    else:
        dot = np.sum(vector1*vector2)
        mag1 = np.sqrt(np.sum(vector1**2))
        mag2 = np.sqrt(np.sum(vector2**2))
        
    cos_angle = dot/(mag1*mag2)
    angle_rad = np.arccos(cos_angle)
    angle = np.degrees(angle_rad)
    return angle 

def cross_product(vector1, vector2):
    """
    This function cross product two vectors.It is a wrap of the numpy cross function
 
    Parameters
    ----------    
    vector1 : ndarray
        array defining the vector(s).
        
    vector2 : ndarray
        array defining the vector(s).
 
    Returns
    -------
    cross_product : ndarray
        numpy array of the cross product
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
       
    cross_product = np.cross(vector1, vector2)
    return cross_product

def dot_product(vector1, vector2):
    """
    This function cross product two vectors. Wrap of numpy dot function
 
    Parameters
    ----------    
    vector1 : ndarray
        array defining the vector.
        
    vector2 : ndarray
        array defining the vector.
 
    Returns
    -------
    dot_product : ndarray
        numpy array of the dot product
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
       
    dot_product = np.dot(vector1, vector2)
    return dot_product

def normalise_vectors(vector_list):
    """
    This function normalise the vectors.
 
    Parameters
    ----------    
    vector_list : 2d ndarray
        numpy array of the vectors.
 
    Returns
    -------
    normalise_vector : ndarray
        normalise vector.
    """
    if type(vector_list) != np.ndarray:
        vector_list = np.array(vector_list)
    naxis = len(vector_list.shape)
    return vector_list/np.linalg.norm(vector_list, ord = 2, axis = naxis-1, keepdims=True)
    
def translate_matrice(tx,ty,tz):
    """
    This function calculate a 4x4 translation matrice.
 
    Parameters
    ----------    
    tx : float
        magnitude of translation in the x-direction.
    
    ty : float
        magnitude of translation in the y-direction.
        
    tz : float
        magnitude of translation in the z-direction.
 
    Returns
    -------
    matrice : ndarray
        4x4 translation matrice.
    """
    mat = np.array([[1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]])
    
    return mat

def scale_matrice(sx,sy,sz):
    """
    This function calculate a 4x4 translation matrice.
 
    Parameters
    ----------    
    sx : float
        scaling in the x-direction.
    
    sy : float
        scaling in the y-direction.
        
    sz : float
        scaling in the z-direction.
 
    Returns
    -------
    matrice : ndarray
        4x4 scaling matrice.
    """
    
    mat = np.array([[sx, 0, 0, 0],
                    [0, sy, 0, 0],
                    [0, 0, sz, 0],
                    [0, 0, 0, 1]])
    return mat

def rotate_matrice(axis, rotation):
    """
    This function calculate a 4x4 translation matrice. The rotation in counter-clockwise.
 
    Parameters
    ----------    
    axis : tuple
        the axis for rotation, must be a normalised vector.
    
    rotation : float
        the rotation in degrees.
 
    Returns
    -------
    matrice : ndarray
        4x4 rotation matrice.
    """
    x = axis[0]
    y = axis[1]
    z = axis[2]
    #convert degree to radian
    rot_rad = math.radians(rotation)
    cos_ang = math.cos(rot_rad)
    sin_ang = math.sin(rot_rad)
    
    r1 = cos_ang + ((x**2)*(1-cos_ang))
    r2 = (x*y*(1-cos_ang)) - (z*sin_ang)
    r3 = (x*z*(1-cos_ang)) + (y*sin_ang)
    r4 = (y*x*(1-cos_ang)) + (z*sin_ang)
    r5 = cos_ang + ((y**2)*(1-cos_ang))
    r6 = (y*z*(1-cos_ang)) - (x*sin_ang)
    r7 = (z*x*(1-cos_ang)) - (y*sin_ang)
    r8 = (z*y*(1-cos_ang)) + (x*sin_ang)
    r9 = cos_ang + ((z**2)*(1-cos_ang))
    
    mat = np.array([[r1, r2, r3, 0],
                    [r4, r5, r6, 0],
                    [r7, r8, r9, 0],
                    [0, 0, 0, 1]])
    return np.round(mat, decimals = 6)

def inverse_matrice(matrice):
    """
    This function calculate a 4x4 translation matrice. The rotation in counter-clockwise.
 
    Parameters
    ----------    
    matrice : ndarray
        4x4 matrice to inverse.
    
    Returns
    -------
    matrice : ndarray
        inverted 4x4 matrice.
    """
    return np.linalg.inv(matrice)
    
def cs2cs_matrice(orig_cs, dest_cs):
    """
    This function calculate a 4x4 matrice for transforming one coordinate to
    another coordinate system.
 
    Parameters
    ----------    
    orig_cs : coordinate system object
        the original coordinate system.
    
    dest_cs : coordinate system object
        the coordinate system to transform to.
 
    Returns
    -------
    matrice : ndarray
        4x4 matrice.
    """
    # #TODO FINISH THE FUNCTION
    # #first move the origin from the original cs to the destination cs
    # orig1 = orig_cs.origin
    # orig2 = dest_cs.origin
    # trsl = orig2 - orig1
    # trsl_mat = translate_matrice(trsl[0],trsl[1],trsl[2])
    # #then get the rotation axis for the y-axis 
    # yd1 = orig_cs.y_dir
    # yd2 = dest_cs.y_dir
    # rot_axis1 = cross_product(yd1, yd2)
    # #need to get the rotation angle
    # rot_angle = angle_btw_2vectors(yd1, yd2)
    # rot_mat = rotate_matrice(rot_axis1, rot_angle)
    # # print(trsl_mat)
    
def face_midxyz(face):
    """
    Calculates the midpt of the face
 
    Parameters
    ----------    
    face : the face object
        the face object.
    
    Returns
    -------
    midxyz : ndarray
        array defining the midpt.
    """
    if face.surface_type == geom.SrfType.POLYGON:
        vertex_list = get.vertices_frm_face(face)
        xyzs = np.array([v.point.xyz for v in vertex_list])
        mid_xyz = xyzs_mean(xyzs)
        
    elif face.surface_type == geom.SrfType.BSPLINE:
        bspline_srf = face.surface
        mid_xyz = np.array(bspline_srf.evaluate_single([0.5,0.5]))
    return mid_xyz

def trsf_xyzs(xyzs, trsf_mat):
    """
    Calculates the transformed xyzs based on the given matrice
 
    Parameters
    ----------    
    xyzs : ndarray
        array defining the points.
    
    Returns
    -------
    trsf_xyzs : ndarray
        the transformed xyzs.
    """
    
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    #add an extra column to the points
    npos = len(xyzs)
    xyzw = np.ones((npos,4))
    xyzw[:,:-1] = xyzs
    t_xyzw = np.dot(xyzw, trsf_mat.T)
    trsf_xyzs = t_xyzw[:,:-1]
    
    return trsf_xyzs

def move_xyzs(xyzs, directions, magnitudes):
    """
    Calculates the moved xyzs based on a direction vectors and magnitudes
 
    Parameters
    ----------    
    xyzs : ndarray
        array defining the points.
    
    directions : ndarray
        array defining the directions.
    
    magnitudes : 1d array
        array defining the magnitude to move.
        
    Returns
    -------
    moved_xyzs : ndarray
        the moved xyzs.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    if type(directions) != np.ndarray:
        directions = np.array(directions)
        
    if type(magnitudes) != np.ndarray:
        magnitudes = np.array(magnitudes)
    
    magnitudes = np.reshape(magnitudes, [len(directions),1])
    moved_xyzs = xyzs + directions*magnitudes
    return moved_xyzs 

def is_anticlockwise(xyzs, ref_vec):
    """
    This function checks if the list of points are arranged anticlockwise in regards to the ref_pyvec. 
    The ref_pyvec must be perpendicular to the points.
 
    Parameters
    ----------
    xyzs : ndarray
        array defining the points.
    
    ref_vec : tuple of floats
        The reference vector must be perpendicular to the list of points. 
        A vec is a tuple that documents the xyz direction of a vector e.g. (x,y,z)
        
    Returns
    -------
    True or False : bool
        If True the list of points are arranged in anticlockwise manner, if False they are not.
    """
    total = [0,0,0]    
    npts = len(xyzs)
    for i in range(npts):
        vec1 = xyzs[i]
        if i == npts-1:
            vec2 = xyzs[0]
        else:
            vec2 = xyzs[i+1]
            
        #cross the two pts
        prod = cross_product(vec1, vec2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    
    res = dot_product(total, ref_vec)
    if res < 0:
        return False 
    else:
        return True

def ray_xyz_tri_intersect(xyz_orig, xyz_dir, xyz1, xyz2, xyz3):
    """
    This function intersect a ray with a triangle
 
    Parameters
    ----------
    xyz_orig : ndarray
        array defining the origin of the ray.
        
    xyz_dir : ndarray
        array defining the ray direction.
    
    xyz_1 : ndarray
        array defining the 1st point of the triangle.
    
    xyz_2 : ndarray
        array defining the 2nd point of the triangle.
    
    xyz_3 : ndarray
        array defining the 3rd point of the triangle.

    Returns
    -------
    intersect_pt : ndarray
        point of intersection. If no intersection returns None
    """
    if type(xyz_orig) != np.ndarray:
        xyz_orig = np.array(xyz_orig)
    
    if type(xyz_dir) != np.ndarray:
        xyz_dir = np.array(xyz_dir)
    
    if type(xyz1) != np.ndarray:
        xyz1 = np.array(xyz1)
    
    if type(xyz2) != np.ndarray:
        xyz2 = np.array(xyz2)
    
    if type(xyz3) != np.ndarray:
        xyz3 = np.array(xyz3)
    
    xyz2_1 = xyz2-xyz1
    xyz3_1 = xyz3-xyz1
    pvec = cross_product(xyz_dir, xyz3_1)
    det = dot_product(xyz2_1, pvec)
    # print('det', det)
    if det < 0.000001:
        return float('-inf')
    
    invDet = 1.0 / det
    tvec = xyz_orig - xyz1
    u = dot_product(tvec, pvec) * invDet
    # print('u', u)
    if u < 0 or u > 1:
        return None
    
    qvec = cross_product(tvec, xyz2_1)
    v = dot_product(xyz_dir,qvec) * invDet
    # print('v', v)
    if v < 0 or u + v > 1:
        return None
    
    magnitude = dot_product(xyz3_1,qvec)*invDet
    dir_magnitude = xyz_dir*magnitude
    intersect_pt = xyz_orig + dir_magnitude 
    return list(intersect_pt)

def rays_xyz_tris_intersect(rays_xyz, tris_xyz):
    """
    This function intersect multiple rays with multiple triangles
 
    Parameters
    ----------
    rays_xyz : ndarray
        array of rays [ray_xyz1,ray_xyz2,...]. Each ray is defined with [[origin], [direction]].
        
    tris_xyz : ndarray
        array of triangles [tri_xyz1, tri_xyz2, ...]. Each triangle is define with [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]].

    Returns
    -------
    intersect_pts : ndarray
        points of intersection. If no intersection returns an empty array
        
    magnitudes : ndarray
        array of magnitude of the rays to the intersection points. If no intersection return an empty array
        
    ray_index : ndarray
        indices of the rays. Corresponds to the intersect pts.
    
    tris_index : ndarray
        indices of the intersected triangles. Corresponds to the intersect pts.
    """
    #============================================
    def dot_product_2dx2d(vectors1, vectors2):
        if type(vectors1) != np.ndarray:
            vectors1 = np.array(vectors1)
        
        if type(vectors2) != np.ndarray:
            vectors2 = np.array(vectors2)
        
        v1sT = vectors1.T
        v2sT = vectors2.T
        v1x = v1sT[0] * v2sT[0]
        v1y = v1sT[1] * v2sT[1]
        v1z = v1sT[2] * v2sT[2]
        
        dp = v1x + v1y+ v1z
        return dp
    
    def acct4obstruction(intPts, mags, rayIndxs, triIndxs):
        indx = np.arange(len(rayIndxs))
        dupIds = utility.id_dup_indices_1dlist(rayIndxs)
        dupIds_flat = list(chain(*dupIds))
        non_obs_indx = utility.find_xs_not_in_ys(indx, dupIds_flat)
        for di in dupIds:
            mag = mags[di]
            index = np.where(mag == mag.min())[0]
            index = index.flatten()
            index = index + di[0]
            non_obs_indx = np.append(non_obs_indx, index)
        
        non_obs_indx.sort()
        nb_intPts = np.take(intPts, non_obs_indx, axis=0)
        nb_mags = np.take(mags, non_obs_indx, axis=0)
        nb_rayIndxs = np.take(rayIndxs, non_obs_indx, axis=0)
        nb_triIndxs = np.take(triIndxs, non_obs_indx, axis=0)
        return nb_intPts, nb_mags, nb_rayIndxs, nb_triIndxs
    #============================================
    dets_threshold = 0.000001
    if type(rays_xyz) != np.ndarray:
        rays_xyz = np.array(rays_xyz)
    
    if type(tris_xyz) != np.ndarray:
        tris_xyz = np.array(tris_xyz)
    
    nrays = len(rays_xyz)
    ntris = len(tris_xyz)
    
    tris_xyz_tiled = np.tile(tris_xyz, (nrays,1,1))
    trisT = np.stack(tris_xyz_tiled, axis=1)
    xyzs1_0 = trisT[1] - trisT[0]
    xyzs2_0 = trisT[2] - trisT[0]
    
    rays_repeat = np.repeat(rays_xyz, ntris, axis=0)
    raysT = np.stack(rays_repeat, axis=1)
    pvecs = cross_product(raysT[1], xyzs2_0)
    dets = dot_product_2dx2d(xyzs1_0, pvecs)
    # print('dets', dets)
    dets_true = np.logical_not(dets<dets_threshold)
    dets_index = np.where(dets_true)
    if len(dets_index) == 0:
        #no intersections at all
        return [],[],[],[]
    #============================================
    dets = np.where(dets_true, dets, np.inf)
    invDet = np.where(dets_true, 1.0/dets, dets)
    invDet = np.where(dets_true, invDet, 0)
    dets_true_repeat = np.repeat(dets_true, 3)
    dets_true_repeat_reshape = np.reshape(dets_true_repeat, (len(raysT[0]),3))
    
    tvecs = np.where(dets_true_repeat_reshape, 
                    raysT[0] - trisT[0], 
                    np.zeros((len(raysT[0]), 3)))
    us = np.where(dets_true, 
                  dot_product_2dx2d(tvecs, pvecs)*invDet, 
                  np.zeros(len(dets_true)))
    # print('us', us)
    us_true = np.logical_and(us>=0, us<=1)
    us_true = np.logical_and(us_true, dets_true)
    us_index = np.where(us_true)
    if len(us_index) == 0:
        #no intersections at all 
        return [],[],[],[]
    #============================================
    us_true_repeat = np.repeat(us_true, 3)
    us_true_repeat_reshape = np.reshape(us_true_repeat, (len(tvecs),3))
    
    qvecs = np.where(us_true_repeat_reshape,
                     cross_product(tvecs, xyzs1_0),
                     np.zeros([len(tvecs),3]))
    vs = np.where(us_true, 
                  dot_product_2dx2d(raysT[1], qvecs)*invDet,
                  np.zeros([len(us_true)]))
    # print('vs', vs)
    vs_true1 = np.logical_not(vs < 0)
    vs_true2 = np.logical_not(us+vs > 1)
    vs_true = np.logical_and(vs_true1, vs_true2)
    vs_true = np.logical_and(vs_true, us_true)
    mags = np.where(vs_true, 
                    dot_product_2dx2d(xyzs2_0, qvecs)*invDet,
                    np.zeros([len(vs_true)]))
    # print('mags',mags)
    mags_reshape = np.reshape(mags,[len(mags),1])
    dir_mags = raysT[1]*mags_reshape
    intPts = raysT[0] + dir_mags
    #============================================
    #separate the intersections into ray-based groups
    #sub zeros with infinity to make processing easier later
    mags = np.where(vs_true, mags, np.full([len(vs_true)], np.inf))
    mags_ray_reshape = np.reshape(mags, (nrays,ntris))
    mags_true = np.logical_not(mags_ray_reshape==np.inf)
    mags_true_index = np.where(mags_true)
    intPts_reshape = np.reshape(intPts, (nrays,ntris,3))
    res_pts = intPts_reshape[mags_true_index]
    res_mag = mags_ray_reshape[mags_true_index]
    res_ray_indx = mags_true_index[0]
    res_tri_indx = mags_true_index[1]
    
    res_pts, res_mag, res_ray_indx, res_tri_indx= acct4obstruction(res_pts, 
                                                                   res_mag, 
                                                                   res_ray_indx, 
                                                                   res_tri_indx)
    
    return res_pts, res_mag, res_ray_indx, res_tri_indx

def rays_faces_intersection(ray_list, face_list):
    """
    This function intersect multiple rays with multiple faces
 
    Parameters
    ----------
    ray_list : list of rays
        array of ray objects
        
    face_list : lsit of faces
        array of face objects
        
    Returns
    -------
    hit_face : array of faces
        faces that are hit with new attributes documenting the intersection pt and the rays that hit it. {'rays_faces_intersection': {'intersection':[], 'ray':[]}}
        
    miss_faces : array of faces
        faces that are not hit by any rays.
        
    hit_rays : array of rays
        rays that hit faces with new attributes documenting the faces and intersections. {'rays_faces_intersection': {'intersection':[], 'hit_face':[]}}
        
    
    miss_rays : array of rays
        rays that did not hit any faces.
    """
    #extract the origin and dir of the rays 
    rays_xyz = [[r.origin,r.dirx] for r in ray_list]
    #need to first triangulate the faces
    tris_xyz = np.array([])
    ntris = 0
    tidx_faces = []
    for f in face_list:
        verts_indx = modify.triangulate_face(f, indices = True)
        verts = verts_indx[0]
        indx = verts_indx[1]
        c_verts = np.take(verts, indx, axis=0)
        tris_xyz = np.append(tris_xyz, c_verts)
        ntri_aface = len(indx)
        tidx_faces.append(np.arange(ntri_aface) + ntris)
        ntris += ntri_aface
    
    #reshape the tris
    tris_xyz = np.reshape(tris_xyz, (ntris, 3, 3))
    res_pts, res_mag, res_ray_indx, res_tri_indx = rays_xyz_tris_intersect(rays_xyz, tris_xyz)
    #now i need to sort it into faces
    hit_faces = []
    miss_faces = []
    for cnt, tidx_face in enumerate(tidx_faces):
        #if the triidx_face are in the hit tri indx, this face is hit by a ray
        hit_true = np.in1d(res_tri_indx, tidx_face)
        hit_indx = np.where(hit_true)[0]
        nhits = len(hit_indx)
        if nhits == 0:
            #this face is not hit by any rays
            miss_faces.append(face_list[cnt])
        elif nhits == 1:
            #this face is hit by one of the ray
            #find the ray and put it in its attributes
            face_ray = ray_list[res_ray_indx[hit_indx[0]]]
            intersect = res_pts[hit_indx[0]]
            hit_face = face_list[cnt]
            hf_attribs = hit_face.attributes
            if 'rays_faces_intersection' in hf_attribs:
                ht_rf_att = hf_attribs['rays_faces_intersection']
                ht_rf_att['intersection'].append(intersect)
                ht_rf_att['ray'].append(face_ray)
            else:
                attribs = {'rays_faces_intersection':
                               { 'intersection': [intersect],
                                'ray': [face_ray]
                                }
                           }
                hit_face.update_attributes(attribs)
            hit_faces.append(hit_face)
            
            #check if the attribs alr exist in the rays
            rattribs = face_ray.attributes
            if 'rays_faces_intersection' in rattribs:
                rays_rf_att = rattribs['rays_faces_intersection']
                rays_rf_att['intersection'].append(intersect)
                rays_rf_att['hit_face'].append(hit_face)
            else:
                rattribs = {'rays_faces_intersection':
                                {'intersection': [intersect],
                                 'hit_face': [hit_face]}
                            }
                face_ray.update_attributes(rattribs)
            # print('1_hit')
        elif nhits > 1:
            #first take a look at the ray index
            #find the rays that are hitting this face
            pts_face = np.take(res_pts, hit_indx, axis=0)
            mags_face = np.take(res_mag, hit_indx, axis=0)
            raysIds_face = np.take(res_ray_indx, hit_indx, axis=0)
            #check if they are instances where a single ray is hitting the same surfaces >1
            idDups = utility.id_dup_indices_1dlist(raysIds_face)
            all_indx = np.arange(len(raysIds_face))
            if len(idDups) > 0:
                # a single ray hits a face more than once
                # the ray is hitting the face at its triangulated seams    
                idDups_flat = list(chain(*idDups))
                nonDup_idxs = utility.find_xs_not_in_ys(all_indx, idDups_flat)
                # double check if hitting at seams is true
                for idDup in idDups:
                    mag = mags_face[idDup]
                    id_magDup = utility.id_dup_indices_1dlist(mag)
                    if len(id_magDup) == 1:    
                        if len(id_magDup[0]) == len(mag):
                            #they are hitting the same spot
                            nonDup_idxs = np.append(nonDup_idxs, idDup[0])
                        else:
                            print('SOMETHING IS WRONG WITH THIS SURFACE')
                    else:
                        print('SOMETHING IS WRONG WITH THIS SURFACE')
                        
            else:
                nonDup_idxs = all_indx
            
            nonDup_idxs.sort()
            raysIds_face = np.take(raysIds_face, nonDup_idxs, axis=0)
            face_rays = np.take(ray_list, raysIds_face, axis=0)
            intersects = np.take(pts_face, nonDup_idxs, axis=0)
            
            hit_face = face_list[cnt]
            hf_attribs = hit_face.attributes
            
            #need to check is it hit by different rays
            if 'rays_faces_intersection' in hf_attribs:
                ht_rf_att = hf_attribs['rays_faces_intersection']
                ht_rf_att['intersection'].extend(intersects)
                ht_rf_att['ray'].extend(face_rays)
            else:
                attribs = {'rays_faces_intersection':
                               { 'intersection': intersects,
                                'ray': face_rays
                                }
                           }
                hit_face.update_attributes(attribs)
                
            hit_faces.append(hit_face)
            
            #check if the attribs alr exist in the rays
            for fcnt,face_ray in enumerate(face_rays):
                rattribs = face_ray.attributes
                if 'rays_faces_intersection' in rattribs:
                    rays_rf_att = rattribs['rays_faces_intersection']
                    rays_rf_att['intersection'].append(intersects[fcnt])
                    rays_rf_att['hit_face'].append(hit_face)
                else:
                    rattribs = {'rays_faces_intersection':
                                    {'intersection': [intersects[fcnt]],
                                     'hit_face': [hit_face]}
                                }
                    face_ray.update_attributes(rattribs)
            # print('multiple hits')
    hit_rays = []
    miss_rays = []
    uq_hit_idx = np.unique(res_ray_indx)
    hit_rays = np.take(ray_list, uq_hit_idx , axis=0)
    rindx = np.arange(len(ray_list))
    miss_true = np.in1d(rindx, uq_hit_idx)
    miss_true = np.logical_not(miss_true)
    miss_idx = np.where(miss_true)[0]
    miss_rays = np.take(ray_list, miss_idx, axis=0)
    return hit_faces, miss_faces, hit_rays, miss_rays
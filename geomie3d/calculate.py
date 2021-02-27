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
import numpy as np

from . import topobj
from . import get

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
    This function cross product two vectors.
 
    Parameters
    ----------    
    vector1 : ndarray
        array defining the vector.
        
    vector2 : ndarray
        array defining the vector.
 
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
    This function cross product two vectors.
 
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
    #first move the origin from the original cs to the destination cs
    orig1 = orig_cs.origin
    orig2 = dest_cs.origin
    trsl = orig2 - orig1
    trsl_mat = translate_matrice(trsl[0],trsl[1],trsl[2])
    #then get the rotation axis for the y-axis 
    yd1 = orig_cs.y_dir
    yd2 = dest_cs.y_dir
    rot_axis1 = cross_product(yd1, yd2)
    #need to get the rotation angle
    rot_angle = angle_btw_2vectors(yd1, yd2)
    rot_mat = rotate_matrice(rot_axis1, rot_angle)
    
    
    #TODO FINISH THE FUNCTION
    # print(trsl_mat)
    
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
    vertex_list = get.vertices_frm_face(face)
    xyzs = np.array([v.point.xyz for v in vertex_list])
    mid_xyz = xyzs_mean(xyzs)
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
        np.array(xyzs)
    
    #add an extra column to the points
    npos = len(xyzs)
    xyzw = np.ones((npos,4))
    xyzw[:,:-1] = xyzs
    t_xyzw = np.dot(xyzw, trsf_mat.T)
    trsf_xyzs = t_xyzw[:,:-1]
    
    return trsf_xyzs

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
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
    This function calculate a 4x4 translation matrice.
 
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
    cos_ang = math.cos(rotation)
    sin_ang = math.sin(rotation)
    
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
    return mat

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
    print(trsl_mat)
    
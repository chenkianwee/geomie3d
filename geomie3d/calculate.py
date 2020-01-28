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
import numpy as np

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
    
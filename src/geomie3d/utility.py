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
import csv
import colorsys
from itertools import chain

import numpy as np

from . import modify
from . import get
from . import create

class CoordinateSystem(object):
    """
    A coordinate system object
    
    Parameters
    ----------
    origin : tuple
        The xyz defining the origin.
        
    x_dir : tuple
        The xyz of a vector defining the x-axis
        
    y_dir : tuple
        The xyz of a vector defining the y-axis  
    
    Attributes
    ----------    
    origin : tuple
        The xyz defining the origin.
        
    x_dir : tuple
        The xyz of a vector defining the x-axis.
        
    y_dir : tuple
        The xyz of a vector defining the y-axis.
    """
    def __init__(self, origin, x_dir, y_dir):
        """Initialises the class"""
        if type(origin) != np.ndarray:
            origin = np.array(origin)
        if type(x_dir) != np.ndarray:
            x_dir = np.array(x_dir)
        if type(y_dir) != np.ndarray:
            y_dir = np.array(y_dir)
            
        self.origin = origin
        self.x_dir = x_dir
        self.y_dir = y_dir

class Ray(object):
    """
    A ray object
    
    Parameters
    ----------
    origin : tuple
        The xyz defining the origin.
        
    dirx : tuple
        The direction of the ray
    
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Attributes
    ----------    
    origin : tuple
        The xyz defining the origin.
        
    dirx : tuple
        The direction of the ray
        
    """
    def __init__(self, origin, dirx, attributes = {}):
        """Initialises the class"""
        if type(origin) != np.ndarray:
            origin = np.array(origin)
        if type(dirx) != np.ndarray:
            dirx = np.array(dirx)
            
        self.origin = origin
        self.dirx = dirx
        self.attributes = attributes
    
    def overwrite_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att 

class Bbox(object):
    """
    A bounding box object
    
    Parameters
    ----------
    bbox_arr : tuple
        Array specifying [minx,miny,minz,maxx,maxy,maxz].
        
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Attributes
    ----------    
    minx : float
        The min x.
    
    miny : float
        The min y.
        
    minz : float
        The min z.
        
    maxx : float
        The max x.
        
    maxy : float
        The max y.
        
    maxz : float
        The max z.
        
    attributes : dictionary
        dictionary of the attributes
        
    """
    def __init__(self, bbox_arr, attributes = {}):
        """Initialises the class"""
        if type(bbox_arr) != np.ndarray:
            bbox_arr = np.array(bbox_arr)
        
        self.bbox_arr = bbox_arr
        self.minx = bbox_arr[0]
        self.miny = bbox_arr[1]
        self.minz = bbox_arr[2]
        self.maxx = bbox_arr[3]
        self.maxy = bbox_arr[4]
        self.maxz = bbox_arr[5]
        self.attributes = attributes
    
    def overwrite_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dictionary
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att

def id_dup_indices_1dlist(lst):
    """
    This function returns numpy array of the indices of the repeating elements in a list.
 
    Parameters
    ----------
    lst : a 1D list
        A 1D list to be analysed.
 
    Returns
    -------
    indices : 2d ndarray
        ndarray of the indices (Nduplicates, indices of each duplicate).
    """
    if type(lst) != np.ndarray:    
        lst = np.array(lst)
        
    idx_sort = np.argsort(lst)
    sorted_lst = lst[idx_sort]
    
    vals, idx_start, count = np.unique(sorted_lst, 
                                       return_counts=True,
                                       return_index=True)

    res = np.split(idx_sort, idx_start[1:])
    vals = vals[count > 1]
    res = list(filter(lambda x: x.size > 1, res))
    return res

def find_xs_not_in_ys(xlst, ylst):
    """
    This function compare the 2 list and find the elements in xlst that is not in ylst.
 
    Parameters
    ----------
    xlst : a 1D list
        The lst to check if the elements exist in the ylst.
    
    ylstt : a 1D list
        The reference lst to compare xlst.
        
    Returns
    -------
    not_in : 1d ndarray
        the elements in xlst and not in ylst.
    """
    if type(xlst) != np.ndarray:    
        xlst = np.array(xlst)
    
    if type(ylst) != np.ndarray:    
        ylst = np.array(ylst)
    
    not_in_true = np.in1d(xlst, ylst)
    not_in_true = np.logical_not(not_in_true)
    not_in_indx = np.where(not_in_true)[0]
    not_in = np.take(xlst, not_in_indx, axis=0)
    return not_in

def separate_dup_non_dup(lst):
    """
    This function return the indices of non dup elements and duplicate elements.
 
    Parameters
    ----------
    lst : a 1D list
        The lst to identify which element is non duplicate, which is duplicate.
        
    Returns
    -------
    indices : ndarray
        ndarray of the indices [[non_dup_idx], [[dup_idx1], [dup_idx2] ... [dup_idxn]]].
    """
    if type(lst) != np.ndarray:    
        lst = np.array(lst)
    
    indx = np.arange(len(lst))
    dupIds = id_dup_indices_1dlist(lst)
    dupIds_flat = list(chain(*dupIds))
    non_dup_indx = find_xs_not_in_ys(indx, dupIds_flat)
    return np.array([non_dup_indx, dupIds], dtype=object)
    
def gen_gridxyz(xrange, yrange, zrange=None):
    """
    This function generate a 2d xy grid.
    
    Parameters
    ----------
    xrange : list of int
        [start, stop, number of intervals].
    
    yrange : list of int
        [start, stop, number of intervals].
    
    zrange : list of int, optional
        [start, stop, number of intervals].
        
    Returns
    -------
    gridxy : 2d array
        2d array with xyz. If zrange == None, xy.
    """
    x = np.linspace(xrange[0], xrange[1], xrange[2])
    y = np.linspace(yrange[0], yrange[1], yrange[2])
    
    if zrange == None:
        xx, yy = np.meshgrid(x,y)
        xx = xx.flatten()
        yy= yy.flatten()
        xys = np.array([xx, yy])
        xys = xys.T
        return xys
    else:
        z = np.linspace(zrange[0], zrange[1], zrange[2])
        xx, yy, zz = np.meshgrid(x,y,z)
        xx = xx.flatten()
        yy= yy.flatten()
        zz = zz.flatten()
        xyzs = np.array([xx, yy, zz])
        xyzs = xyzs.T
        return xyzs

def pseudocolor(val, minval, maxval, inverse = False):
    """
    This function converts a value into a rgb value with reference to the minimum and maximum value.
 
    Parameters
    ----------
    val : float
        The value to be converted into rgb.
        
    minval : float
        The minimum value of the falsecolour rgb.
        
    maxval : float
        The maximum value of the falsecolour rgb.
    
    inverse : bool
        False for red being max, True for blue being maximum.
        
    Returns
    -------
    rgb value : tuple of floats
        The converted rgb value.
    """
    # convert val in range minval..maxval to the range 0..120 degrees which
    # correspond to the colors red..green in the HSV colorspace
    if val <= minval:
        if inverse == False:
            h = 250.0
        else:
            h=0.0
    elif val>=maxval:
        if inverse == False:
            h = 0.0
        else:
            h=250.0
    else:
        if inverse == False:
            h = 250 - (((float(val-minval)) / (float(maxval-minval)))*250)
        else:
            h = (((float(val-minval)) / (float(maxval-minval)))*250)
    # convert hsv color (h,1,1) to its rgb equivalent
    # note: the hsv_to_rgb() function expects h to be in the range 0..1 not 0..360
    r, g, b = colorsys.hsv_to_rgb(h/360, 1., 1.)
    return r, g, b
    
def calc_falsecolour(vals, minval, maxval, inverse = False):
    """
    This function converts a list of values into a list of rgb values with reference to the minimum and maximum value.
 
    Parameters
    ----------
    vals : list of float
        A list of values to be converted into rgb.
        
    minval : float
        The minimum value of the falsecolour rgb.
        
    maxval : float
        The maximum value of the falsecolour rgb.
    
    inverse : bool
        False for red being max, True for blue being maximum.
        
    Returns
    -------
    rgb value : list of tuple of floats
        The converted list of rgb value.
    """
    res_colours = []
    for result in vals:
        r,g,b = pseudocolor(result, minval, maxval, inverse=inverse)
        colour = (r, g, b)
        res_colours.append(colour)
    return res_colours

def rgb2val(rgb, minval, maxval):
    """
    This function converts a rgb of value into its original value with reference to the minimum and maximum value.
 
    Parameters
    ----------
    rgb : tuple of floats
        The rgb value to be converted.
        
    minval : float
        The minimum value of the falsecolour rgb.
        
    maxval : float
        The maximum value of the falsecolour rgb.
        
    Returns
    -------
    original value : float
        The orignal float value.
    """
    hsv = colorsys.rgb_to_hsv(rgb[0],rgb[1],rgb[2])
    y = hsv[0]*360
    orig_val_part1 = ((-1*y) + 250)/250.0
    orig_val_part2 = maxval-minval
    orig_val = (orig_val_part1*orig_val_part2)+minval
    return orig_val

def write2ply(topo_list, ply_path, square_face = False):
    """
    Writes the topologies to a ply file. only works for points and face
 
    Parameters
    ----------
    topo_list : list of topo objects
        Topos to be written to ply.
        
    ply_path : str
        Path to write to.
        
    square_face : bool, optional
        If you know all the faces have 4 vertices and you want to preserve that turn this to True.
        
    """
    header = ['ply\n', 'format ascii 1.0\n', 'element vertex', 
              'property float32 x\n', 'property float32 y\n', 
              'property float32 z\n', 'element face',
              'property list uint8 int32 vertex_index\n', 'end_header\n']
    
    cmp = create.composite(topo_list)
    sorted_d = get.unpack_composite(cmp)
    #=================================================================================
    #get all the topology that can be viz as mesh
    #=================================================================================
    all_faces = []
    faces = sorted_d['face']
    if len(faces) > 0:
        all_faces = faces
    
    shells = sorted_d['shell']
    if len(shells) > 0:
        shells2faces = np.array([get.faces_frm_shell(shell) for shell in shells])
        shells2faces = shells2faces.flatten()
        all_faces = np.append(all_faces, shells2faces)
    
    solids = sorted_d['solid']
    if len(solids) > 0:
        solids2faces = np.array([get.faces_frm_solid(solid) for solid in solids])
        solids2faces = solids2faces.flatten()
        all_faces = np.append(all_faces, solids2faces)
    
    nverts = 0
    verts = []
    idxs = []
    #if there are faces to be viz
    if len(all_faces) > 0:
        if square_face == True:
            f_v = []
            for f in all_faces:
                f_v.extend(get.vertices_frm_face(f).tolist())

            verts = [v.point.xyz.tolist() for v in f_v]
            idxs = np.arange(len(verts))
            idxs = np.reshape(idxs, [len(all_faces), 4])
        else:
            mesh_dict = modify.faces2mesh(all_faces)
            verts = mesh_dict['vertices']
            idxs = mesh_dict['indices']
            #flip the vertices to be clockwise
            idxs = np.flip(idxs, axis=1)
            
    #=================================================================================
    #get all the topology that can viz as points
    #=================================================================================
    vertices = sorted_d['vertex']
    if len(vertices) > 0:
        points_vertices = [v.point.xyz.tolist() for v in vertices]
        verts.extend(points_vertices)
        
    nverts = len(verts)
    header[2] = 'element vertex ' + str(nverts) + '\n'
    nfaces = len(idxs)
    header[6] = 'element face ' + str(nfaces) + '\n'

    for xyz in verts:
        v_str = str(xyz[0]) + ' ' + str(xyz[1]) + ' ' + str(xyz[2]) + '\n'
        header.append(v_str)
    if square_face == True:
        for idx in idxs:
            i_str = '4 ' + str(idx[0]) + ' ' + str(idx[1]) + ' ' + str(idx[2]) + ' ' + str(idx[3]) + '\n'
            header.append(i_str)
    else:
        for idx in idxs:
            i_str = '3 ' + str(idx[0]) + ' ' + str(idx[1]) + ' ' + str(idx[2]) + '\n'
            header.append(i_str)
        
    f = open(ply_path, "w")
    f.writelines(header)
    f.close()
    
def write2pts(vertex_list, pts_path):
    """
    Writes the vetices to a pts file. only works for vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology. 
        
    pts_path : str
        Path to write to.
        
    """
    xyz_ls = []
    for v in vertex_list:
        xyz = v.point.xyz
        normal = [0,0,0]
        att = v.attributes
        if 'normal' in att:
            normal = att['normal']
        v_str = str(xyz[0]) + ',' + str(xyz[1]) + ',' + str(xyz[2]) + ',' + str(normal[0]) + ',' + str(normal[1]) + ',' + str(normal[2]) + '\n'
        xyz_ls.append(v_str)
    
    f = open(pts_path, "w")
    f.writelines(xyz_ls)
    f.close()
    
def write2csv(rows2d, csv_path, mode = 'w'):
    """
    Writes the rows to a csv file.
 
    Parameters
    ----------
    rows2d : a 2d list
        A 2d list to write to csv. e.g. [['time', 'datax', 'datay'],['2022-02-02', 10, 20]]
        
    csv_path : str
        Path to write to.
    """
    # writing to csv file 
    with open(csv_path, mode, newline='') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the data rows 
        csvwriter.writerows(rows2d)

def read_csv(csv_path):
    """
    read the csv file.
 
    Parameters
    ----------
    csv_path : str
        Path to write to.
        
    Returns
    -------
    lines : list of lines
        list of rows of the csv content.
    """
    with open(csv_path, mode ='r')as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        
    # displaying the contents of the CSV file
    csvFile = list(csvFile)
    return csvFile
    
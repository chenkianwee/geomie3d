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
import math
import json
import colorsys
from itertools import chain

import numpy as np

from . import modify
from . import get
from . import create
from . import topobj
from . import calculate

class Bbox(object):
    def __init__(self, bbox_arr: list[float], attributes: dict = {}):
        """
        A bounding box object
        
        Parameters
        ----------
        bbox_arr : list[float]
            Array specifying [minx, miny, minz, maxx, maxy, maxz].
            
        attributes : dict, optional
            dictionary of the attributes.
        """
        if type(bbox_arr) != np.ndarray:
            bbox_arr = np.array(bbox_arr)
        
        self.bbox_arr: np.ndarray = bbox_arr
        """Array of [minx, miny, minz, maxx, maxy, maxz]"""
        self.minx: float = bbox_arr[0]
        """The min x"""
        self.miny: float = bbox_arr[1]
        """The min y"""
        self.minz: float = bbox_arr[2]
        """The min z"""
        self.maxx: float = bbox_arr[3]
        """The max x"""
        self.maxy: float = bbox_arr[4]
        """The max y"""
        self.maxz: float = bbox_arr[5]
        """The max z"""
        self.attributes: dict = attributes
        """dictionary of the attributes"""
    
    def overwrite_attributes(self, new_attributes: dict):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dict
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes: dict):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dict
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att

class CoordinateSystem(object):
    def __init__(self, origin: list[float], x_dir: list[float], y_dir:list[float]):
        """A coordinate system object
    
        Parameters
        ----------
        origin : list[float]
            The xyz defining the origin.
            
        x_dir : list[float]
            The xyz of a vector defining the x-axis
            
        y_dir : list[float]
            The xyz of a vector defining the y-axis
        """
        if type(origin) != np.ndarray:
            origin = np.array(origin)
        if type(x_dir) != np.ndarray:
            x_dir = np.array(x_dir)
        if type(y_dir) != np.ndarray:
            y_dir = np.array(y_dir)
            
        self.origin: np.ndarray = origin
        """The xyz defining the origin"""
        self.x_dir: np.ndarray = x_dir
        """The xyz of a vector defining the x-axis"""
        self.y_dir: np.ndarray = y_dir
        """The xyz of a vector defining the y-axis"""

class Plane(object):
    def __init__(self, pointxyz: list[float], nrmlxyz: list[float], attributes: dict = {}):
        """
        A plane object is define as ax + by + cz + d = 0. http://mathonline.wikidot.com/point-normal-form-of-a-plane
        
        Parameters
        ----------
        pointxyz : list[float]
            a point on the plane.

        nrmlxyz : list[float]
            the normal of the plane.
            
        attributes : dict, optional
            dictionary of the attributes.
        """
        self.a: float = nrmlxyz[0]
        """the a coefficient"""
        self.b: float = nrmlxyz[1]
        """the b coefficient"""
        self.c: float = nrmlxyz[2]
        """the c coefficient"""
        self.attributes: dict = attributes
        """dictionary of the attributes"""
        self.d: float = self.calc_d(pointxyz)
        """the d coefficient"""

    def calc_d(self, pointxyz: list[float]) -> float:
        """
        Calculates the d coefficient of the plane.
     
        Parameters
        ----------
        pointxyz: list[float]
            a point on the plane.

        Returns
        -------
        float
            the coefficient d.
        """
        d = calculate.planes_frm_pointxyzs_nrmls([pointxyz], [[self.a, self.b, self.c]])[0][3]
        return d
    
    def overwrite_attributes(self, new_attributes: dict):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dict
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes: dict):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dict
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att

class Ray(object):
    def __init__(self, origin: list[float], dirx: list[float], attributes: dict = {}):
        """
        A ray object
        
        Parameters
        ----------
        origin : list[float]
            The xyz defining the origin.
            
        dirx : list[float]
            The direction of the ray
        
        attributes : dict, optional
            dictionary of the attributes.

        """
        if type(origin) != np.ndarray:
            origin = np.array(origin)
        if type(dirx) != np.ndarray:
            dirx = np.array(dirx)
            
        self.origin: np.ndarray = origin
        """The xyz defining the origin"""
        self.dirx: np.ndarray = dirx
        """The direction of the ray"""
        self.attributes: dict = attributes
        """dictionary of the attributes"""
    
    def overwrite_attributes(self, new_attributes: dict):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dict
            The dictionary of attributes appended to the object.
        """
        self.attributes = new_attributes
        
    def update_attributes(self, new_attributes: dict):
        """
        This function overwrites the attribute dictionary with the new dictionary.
     
        Parameters
        ----------
        new_attributes : dict
            The dictionary of attributes appended to the object.
        """
        old_att = self.attributes
        update_att = old_att.copy()
        update_att.update(new_attributes)
        self.attributes = update_att 

def calc_falsecolour(vals: list[float], minval: float, maxval: float, inverse: bool = False) -> list[list[float]]:
    """
    This function converts a list of values into a list of rgb values with reference to the minimum and maximum value.
 
    Parameters
    ----------
    vals : list[float]
        A list of values to be converted into rgb.
        
    minval : float
        The minimum value of the falsecolour rgb.
        
    maxval : float
        The maximum value of the falsecolour rgb.
    
    inverse : bool
        False for red being max, True for blue being maximum.
        
    Returns
    -------
    rgb value : list[list[float]]
        The converted list of rgb value.
    """
    res_colours = []
    for result in vals:
        r,g,b = pseudocolor(result, minval, maxval, inverse=inverse)
        colour = (r, g, b)
        res_colours.append(colour)
    return res_colours

def check_2dlist_is_hmgnz(lst2d: list[list[list]] | np.ndarray) -> bool:
    """
    Check if the list is homogeneous.
 
    Parameters
    ----------
    lst : list | np.ndarray
        The list to check.
        
    Returns
    -------
    is_hmgnz : bool
        True if list is homogeneous.
    """
    each_set_cnt = []
    for setx in lst2d:
        each_set_cnt.append(len(setx))

    each_set_cnt = np.array(each_set_cnt)
    uniq = np.unique(each_set_cnt)
    if len(uniq) == 1:
        return True
    else:
        return False

def find_xs_in_ys(xlst: list, ylst: list) -> np.ndarray:
    """
    This function compare the 2 list and find the elements in xlst that is in ylst.
 
    Parameters
    ----------
    xlst : list
        The lst to check if the elements exist in the ylst.
    
    ylstt : list
        The reference lst to compare xlst.
        
    Returns
    -------
    in : np.ndarray
        the elements in xlst that is in ylst.
    """
    if type(xlst) != np.ndarray:    
        xlst = np.array(xlst)
    
    if type(ylst) != np.ndarray:    
        ylst = np.array(ylst)
    
    in_true = np.isin(xlst, ylst)
    in_indx = np.where(in_true)[0]
    inx = np.take(xlst, in_indx, axis=0)
    return inx

def find_xs_not_in_ys(xlst: list, ylst: list) -> np.ndarray:
    """
    This function compare the 2 list and find the elements in xlst that is not in ylst.
 
    Parameters
    ----------
    xlst : list
        The lst to check if the elements exist in the ylst.
    
    ylstt : list
        The reference lst to compare xlst.
        
    Returns
    -------
    not_in : np.ndarray
        the elements in xlst and not in ylst.
    """
    if type(xlst) != np.ndarray:    
        xlst = np.array(xlst)
    
    if type(ylst) != np.ndarray:    
        ylst = np.array(ylst)
    
    not_in_true = np.isin(xlst, ylst)
    not_in_true = np.logical_not(not_in_true)
    not_in_indx = np.where(not_in_true)[0]
    not_in = np.take(xlst, not_in_indx, axis=0)
    return not_in

def gen_gridxyz(xrange: list[int], yrange: list[int], zrange: list[int] = None) -> np.ndarray:
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
    gridxy : np.ndarray
        np.ndarray(shape(number of grid pts, 3)), if zrange is specified: np.ndarray(shape(number of grid pts, 3)).
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
        yy, zz, xx = np.meshgrid(y,z,x)
        xx = xx.flatten()
        yy= yy.flatten()
        zz = zz.flatten()
        xyzs = np.array([xx, yy, zz])
        xyzs = xyzs.T
        return xyzs

def id_dup_indices_1dlist(lst: list) -> list[list]:
    """
    This function returns numpy array of the indices of the repeating elements in a list.
 
    Parameters
    ----------
    lst : list
        A 1D list to be analysed.
 
    Returns
    -------
    indices : list[list]
        list[list(shape(Nduplicates, indices of each duplicate))].
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

def pseudocolor(val: float, minval: float, maxval: float, inverse: bool = False) -> list[float]:
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
    rgb value : list[float]
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

def read_csv(csv_path: str) -> list:
    """
    read the csv file.
 
    Parameters
    ----------
    csv_path : str
        Path to write to.
        
    Returns
    -------
    lines : list
        list of rows of the csv content.
    """
    with open(csv_path, mode ='r')as file:
        # reading the CSV file
        csvFile = csv.reader(file)
        
    # displaying the contents of the CSV file
    csvFile = list(csvFile)
    return csvFile

def read_geomie3d(path: str) -> dict:
    """
    reads the topologies to a geomie3d file.
 
    Parameters
    ----------
    path : str
        file to read
    
    Returns
    -------
    model_dictionary : dict
        dictionary with keywords 'topo_list' and 'attributes'.
    """
    with open(path, 'r') as f:
        data = json.load(f)
    
    model = data['model']
    topo_list = []
    for topod in model:
        if topod['topo_type'] == 0:
            topo = _read_vertex(topod)
            topo_list.append(topo)
        elif topod['topo_type'] == 1:
            topo = _read_edge(topod)
            topo_list.append(topo)
        elif topod['topo_type'] == 2:
            topo = _read_wire(topod)
            topo_list.append(topo)
        elif topod['topo_type'] == 3:
            topo = _read_face(topod)
            topo_list.append(topo)
        elif topod['topo_type'] == 4:
            topo = _read_shell(topod)
            topo_list.append(topo)
        elif topod['topo_type'] == 5:
            topo = _read_solid(topod)
            topo_list.append(topo)
        elif topod['topo_type'] == 6:
            topo = _read_composite(topod)
            topo_list.append(topo)
             
    return {'topo_list': topo_list, 'attributes': data['attributes']}

def rgb2val(rgb: list[float], minval: float, maxval: float) -> float:
    """
    This function converts a rgb of value into its original value with reference to the minimum and maximum value.
 
    Parameters
    ----------
    rgb : list[float]
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

def separate_dup_non_dup(lst: list) -> list:
    """
    This function return the indices of non dup elements and duplicate elements.
 
    Parameters
    ----------
    lst : list
        The lst to identify which element is non duplicate, which is duplicate.
        
    Returns
    -------
    indices : list
        list of indices [[non_dup_idx], [[dup_idx1], [dup_idx2] ... [dup_idxn]]].
    """
    if type(lst) != np.ndarray:    
        lst = np.array(lst)
    
    indx = np.arange(len(lst))
    dupIds = id_dup_indices_1dlist(lst)
    dupIds_flat = list(chain(*dupIds))
    non_dup_indx = find_xs_not_in_ys(indx, dupIds_flat)
    # return np.array([non_dup_indx, dupIds], dtype=object)
    return [non_dup_indx, dupIds]
    
def viz1axis_timeseries(data_dict_ls: list[dict], plot_title: str, xaxis_label: str, yaxis_label: str, filepath: str,
                        yaxis_lim: list[float] = None, dateformat: str = None, xtick_rot: float = 40, label_fontsize: int = 18, tick_fontsize: int = 16,
                        legend_loc: dict = None, tight_layout: bool = True, inf_lines: list[dict] = None, regions: list[dict] = None, viz: bool = True):
    """
    Viz timeseries data in a 1 axis x-y plot
    
    Parameters
    ----------
    data_dict_ls : list[dict]
        dictionary with the following keys.
        - 'datax', list of timestamps of the data
        - 'datay', list of y-values of the data
        - 'linestyle', tuple of str, 'solid', 'dotted', 'dashed', 'dashdot', (0, (1, 5, 5, 5))
        - 'linewidth', float, width of the line
        - 'marker', str, empty string for no markers '',the markers viz of the data set, https://matplotlib.org/stable/api/markers_api.html
        - 'marker_size', int, the size of the marker
        - 'color', str, color of the data set viz, https://matplotlib.org/stable/tutorials/colors/colors.html
        - 'label', str the label of the data set in the legend
    
    plot_title : str
        tite of the plot
    
    xaxis_label : str
        label of the xaxis
    
    yaxis_label : str
        label of the yaxis
        
    filepath : str
        filepath to save the generated graph.
    
    yaxis_lim : list[float], optional
        tuple specifying the lower and upper limit of the yaxis (lwr, uppr)
    
    dateformat : str, optional
        specify how to viz the date time on the x-axis, e.g.'%Y-%m-%dT%H:%M:%S'. Default 'H:%M:%S'

    xtick_rot : float, optional
        specify the rotation of the placement of xticks. Default 40 degree 

    label_fontsize : int, optional
        the fontsize of the labels, include x and y label and the title. Default 18 

    tick_fontsize : int, optional
        the fontsize of the ticks, include x and y ticks. Default 16
    
    legend_loc : dict, optional
        specify the location of the legend
        - 'loc', str, e.g. upper right, center left, lower center
        - 'bbox_to_anchor',tuple, (x, y)
        - 'ncol', int, number of columns for the legend box
        
    tight_layout : bool, optional
        turn on tight layout. default True
        
    inf_lines : list[dict], optional
        dictionary describing the infinite line.
        - label: str to describe the infinite line
        - angle: float, the angle of the infinite line, 0=horizontal, 90=vertical
        - pos: float, for horizontal and vertical line a single value is sufficient, for slanted line two points (x1,y1) and (x2,y2) is required.
        - colour: tuple, (r,g,b,a)
    
    regions : list[dict], optional
        dictionary describing the region on the graph.
        - label: str to describe the region.
        - orientation: str, 'vertical', 'horizontal' or 'custom'
        - range: list of floats, [lwr_limit, upr_limit] e.g. [50,70], if 'custom', [[xrange], [yrange1], [yrange2]]
        - colour: tuple, (r,g,b,a)

    viz : bool, optional
        if set to True will show the graph. Default to True.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    label_fontsize = 14
    tick_fontsize = 12
    fig, ax1 = plt.subplots()
    plt.title(plot_title, fontsize = label_fontsize )
    ax1.set_xlabel(xaxis_label, fontsize = label_fontsize)
    ax1.set_ylabel(yaxis_label, fontsize = label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    if yaxis_lim is not None:
        plt.ylim(yaxis_lim[0],yaxis_lim[1])
    
    y = []
    x = []
    for dd in data_dict_ls:
        x1 = list(dd['datax'])
        y1 = list(dd['datay'])
        y.extend(y1)
        x.extend(x1)
        ax1.plot(x1, y1, linestyle=dd['linestyle'], linewidth=dd['linewidth'], marker = dd['marker'], 
                 markersize = dd['marker_size'], 
                 c = dd['color'], label = dd['label'])
    
    if dateformat is None:
        dateformat = '%H:%M:%S'
        
    formatter = DateFormatter(dateformat)
    ax1.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    for label in ax1.get_xticklabels():
        label.set_rotation(xtick_rot)
    
    if inf_lines is not None:
        for l in inf_lines:
            if l['angle'] == 0:
                ax1.axhline(y=l['pos'], color=l['colour'], linestyle='--', label = l['label'])
            elif l['angle'] == 90:
                ax1.axvline(x=l['pos'], color=l['colour'], linestyle='--', label = l['label'])
            else:
                ax1.axline(l['pos'], slope=math.tan(l['angle']),  color=l['colour'], linestyle='--', label = l['label'])
    
    if regions is not None:
        for r in regions:
            if r['orientation'] == 'vertical':
                if yaxis_lim is not None:
                    ax1.fill_betweenx([yaxis_lim[0],yaxis_lim[1]], r['range'][0], r['range'][1], label = r['label'], color = r['colour'])
                else:
                    ax1.fill_betweenx([min(y), max(y)], r['range'][0], r['range'][1], label = r['label'], color = r['colour'])
                    
            elif r['orientation'] == 'horizontal':
                ax1.fill_between([min(x), max(x)], r['range'][0], r['range'][1], label = r['label'], color = r['colour'])
            
            elif r['orientation'] == 'custom':
                ax1.fill_between(r['range'][0], r['range'][1], r['range'][2], label = r['label'], color = r['colour'])
    
    if legend_loc is not None:
        ax1.legend(loc=legend_loc['loc'], 
                   bbox_to_anchor=legend_loc['bbox_to_anchor'], 
                   fancybox=True, ncol=legend_loc['ncol'], fontsize = tick_fontsize)
    
    ax1.grid(True, axis = 'y', linestyle='--', linewidth = 0.3)
    ax1.grid(True, axis = 'x', linestyle='--', linewidth = 0.3)
    
    if tight_layout==True:
        plt.tight_layout()
    
    plt.savefig(filepath, bbox_inches = "tight", dpi = 300, transparent=False)
    plt.close()
    if viz == True:
        plt.show()
    
def viz2axis_timeseries(y1_data_dict_ls: list[dict], y2_data_dict_ls: list[dict], plot_title: str, xaxis_label: str, yaxis1_label: str,
                        yaxis2_label: str, filepath: str, yaxis1_lim: list[float] = None, yaxis2_lim: list[float] = None, yaxis2_color: str = 'b',
                        dateformat: str = None, xtick_rot: float = 40, label_fontsize: int = 18, tick_fontsize: int = 16, legend_loc1: dict = None, legend_loc2: dict = None, tight_layout: bool = True, 
                        inf_lines: list[dict] = None, regions: list[dict] = None, viz: bool = True):
    """
    Viz timeseries data in a 1 axis x-y plot

    Parameters
    ----------
    y1_data_dict_ls : list[dict]
        dictionary with the following keys.
        - 'datax', list of timestamps of the data
        - 'datay', list of y-values of the data
        - 'linestyle', tuple of str, 'solid', 'dotted', 'dashed', 'dashdot', (0, (1, 5, 5, 5))
        - 'linewidth', float, width of the line
        - 'marker', str, '' for no markers, the markers viz of the data set, https://matplotlib.org/stable/api/markers_api.html
        - 'marker_size', int, the size of the marker
        - 'color', str, color of the data set viz, https://matplotlib.org/stable/tutorials/colors/colors.html
        - 'label', str the label of the data set in the legend

    y2_data_dict_ls : list[dict]
        dictionary with the same keys as y1_data_dict_ls. Data here will be viz on y2 axis.

    plot_title : str
        tite of the plot

    xaxis_label : str
        label of the xaxis

    yaxis1_label : str
        label of the yaxis1

    yaxis2_label : str
        label of the yaxis2

    filepath : str
        filepath to save the generated graph.

    yaxis1_lim : list[float], optional
        tuple specifying the lower and upper limit of the yaxis1 (lwr, uppr)

    yaxis2_lim : list[float], optional
        tuple specifying the lower and upper limit of the yaxis2 (lwr, uppr)
        
    yaxis2_color : str, optional
        str specifying the color of the second yaxis, default is 'b' blue

    dateformat : str, optional
        specify how to viz the date time on the x-axis, e.g.'%Y-%m-%dT%H:%M:%S'. Default 'H:%M:%S'

    xtick_rot : float, optional
        specify the rotation of the placement of xticks. Default 40 degree

    label_fontsize : int, optional
        the fontsize of the labels, include x and y label and the title. Default 18 

    tick_fontsize : int, optional
        the fontsize of the ticks, include x and y ticks. Default 16
        
    legend_loc1 : dict, optional
        specify the location of the legend of the first y-axis. If None no legend.
        - 'loc', str, e.g. upper right, center left, lower center, best
        - 'bbox_to_anchor',tuple, (x, y)
        - 'ncol', int, number of columns for the legend box

    legend_loc2 : dict, optional
        specify the location of the legend of the 2nd y-axis. If None no legend
        - 'loc', str, e.g. upper right, center left, lower center, best
        - 'bbox_to_anchor',tuple, (x, y)
        - 'ncol', int, number of columns for the legend box

    tight_layout : bool, optional
        turn on tight layout. default True
        
    inf_lines : list[dict], optional
        dictionary describing the infinite line.
        - label: str to describe the infinite line
        - angle: float, the angle of the infinite line, 0=horizontal, 90=vertical
        - pos: float, for horizontal and vertical line a single value is sufficient, for slanted line two points (x1,y1) and (x2,y2) is required.
        - colour: tuple, (r,g,b,a)
    
    regions : list[dict], optional
        dictionary describing the region on the graph.
        - label: str to describe the region.
        - orientation: str, vertical or horizontal
        - range: list of floats, [lwr_limit, upr_limit] e.g. [50,70] 
        - colour: tuple, (r,g,b,a)

    viz : bool, optional
        if set to True will show the graph. Default to True.
    """
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    fig, ax1 = plt.subplots()
    plt.title(plot_title, fontsize=label_fontsize)
    ax1.set_xlabel(xaxis_label, fontsize=label_fontsize)
    ax1.set_ylabel(yaxis1_label, fontsize=label_fontsize)
    plt.xticks(fontsize=tick_fontsize)
    ax1.tick_params(axis='y', labelsize=tick_fontsize)
    if yaxis1_lim is not None:
        plt.ylim(yaxis1_lim[0],yaxis1_lim[1])
    
    y = []
    x = []
    for dd in y1_data_dict_ls:
        x1 = list(dd['datax'])
        y1 = list(dd['datay'])
        y.extend(y1)
        x.extend(x1)
        ax1.plot(x1, y1,marker = dd['marker'], markersize = dd['marker_size'],
                 linestyle = dd['linestyle'], linewidth=dd['linewidth'],
                 c = dd['color'], label = dd['label'])

    if dateformat is None:
        dateformat = '%H:%M:%S'

    formatter = DateFormatter(dateformat)
    ax1.xaxis.set_major_formatter(formatter)
    plt.gcf().autofmt_xdate()
    for label in ax1.get_xticklabels():
        label.set_rotation(xtick_rot)
    
    if inf_lines is not None:
        for l in inf_lines:
            if l['angle'] == 0:
                ax1.axhline(y=l['pos'], color=l['colour'], linestyle='--', label = l['label'])
            elif l['angle'] == 90:
                ax1.axvline(x=l['pos'], color=l['colour'], linestyle='--', label = l['label'])
            else:
                ax1.axline(l['pos'], slope=math.tan(l['angle']),  color=l['colour'], linestyle='--', label = l['label'])
    
    if regions is not None:
        for r in regions:
            if r['orientation'] == 'vertical':
                if yaxis1_lim is not None:
                    ax1.fill_betweenx([yaxis1_lim[0],yaxis1_lim[1]], r['range'][0], r['range'][1], label = r['label'], color = r['colour'])
                else:
                    ax1.fill_betweenx([min(y), max(y)], r['range'][0], r['range'][1], label = r['label'], color = r['colour'])
            elif r['orientation'] == 'horizontal':
                ax1.fill_between([min(x), max(x)], r['range'][0], r['range'][1], label = r['label'], color = r['colour'])
    
    if legend_loc1 is not None:
        ax1.legend(loc=legend_loc1['loc'],
                   bbox_to_anchor=legend_loc1['bbox_to_anchor'],
                   fancybox=True, ncol=legend_loc1['ncol'], fontsize = tick_fontsize)

    ax1.grid(True, axis = 'y', linestyle='--', linewidth = 0.3)
    ax1.grid(True, axis = 'x', linestyle='--', linewidth = 0.3)

    #=================================================================================================================================
    #THE SECOND AXIS
    #=================================================================================================================================
    ax2 = ax1.twinx()
    ax2.tick_params(axis='y', labelsize=tick_fontsize)
    for dd in y2_data_dict_ls:
        ax2.plot(list(dd['datax']), list(dd['datay']),
                  linestyle = dd['linestyle'], linewidth=dd['linewidth'],
                  marker = dd['marker'], markersize = dd['marker_size'],
                  c = dd['color'], label = dd['label'])

    ax2.set_ylabel(yaxis2_label, color='b', fontsize=label_fontsize)
    ax2.tick_params('y', colors = yaxis2_color)
    if yaxis2_lim is not None:
        ax2.set_ylim(yaxis2_lim[0],yaxis2_lim[1])

    if legend_loc2 is not None:
        ax2.legend(loc=legend_loc2['loc'],
                   bbox_to_anchor=legend_loc2['bbox_to_anchor'],
                   fancybox=True, ncol=legend_loc2['ncol'], fontsize = tick_fontsize)

    if tight_layout == True:
      plt.tight_layout()

    plt.savefig(filepath, bbox_inches = "tight", dpi = 300, transparent=False)
    plt.close()
    if viz == True:
        plt.show()

def write2csv(rows2d: list[list], csv_path: str, mode: str = 'w'):
    """
    Writes the rows to a csv file.
 
    Parameters
    ----------
    rows2d : list[list]
        A 2d list to write to csv. e.g. [['time', 'datax', 'datay'],['2022-02-02', 10, 20]]
        
    csv_path : str
        Path to write to.
    
    mode : str, optional
        Mode of writing. default = 'w', can be 'w', 'a' = append
    """
    # writing to csv file 
    with open(csv_path, mode, newline='') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile) 
        # writing the data rows 
        csvwriter.writerows(rows2d)

def write2geomie3d(topo_list: list[topobj.Topology], res_path: str, attributes: dict = {}):
    """
    Writes the topologies to a geomie3d file.
 
    Parameters
    ----------
    topo_list : list[topobj.Topology]
        Topos to be written to .geomie3d.
        
    res_path : str
        Path to write to.
    
    attributes : dict, optional
        attributes of the whole model.
        
    """
    model = {'model': [topo.to_dict() for topo in topo_list], 'attributes': attributes}
    with open(res_path, 'w') as f:
        json.dump(model, f)

def write2ply(topo_list: list[topobj.Topology], ply_path: str, square_face: bool = False):
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

def write2pts(vertex_list: list[topobj.Vertex], pts_path: str):
    """
    Writes the vetices to a pts file. only works for vertices.
 
    Parameters
    ----------
    vertex_list : list[topobj.Vertex]
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

def _read_vertex(vertex_dict: dict) -> topobj.Vertex:
    """
    reads the vertex of a geomie3d file.
 
    Parameters
    ----------
    vertex_dict : dict
        dictionary of the vertex from the file. Refer to the Vertex object for the schema
    
    Returns
    -------
    vertex : topobj.Vertex
        the vertex object.
    """
    v = create.vertex(vertex_dict['point'], attributes = vertex_dict['attributes'])
    return v

def _read_edge(edge_dict: dict) -> topobj.Edge:
    """
    reads the edge of a geomie3d file.
 
    Parameters
    ----------
    edge_dict : dict
        dictionary of the edge from the file. Refer to the edge object for the schema
    
    Returns
    -------
    edge : topobj.Edge
        the edge object.
    """
    crv_type = edge_dict['curve_type']
    if crv_type == 0:
        vlist = create.vertex_list(edge_dict['vertex_list'])
        e = create.pline_edge_frm_verts(vlist, attributes = edge_dict['attributes'])
    elif crv_type == 1:
        e = create.bspline_edge_frm_xyzs(edge_dict['ctrlpts'], degree = edge_dict['degree'], resolution = edge_dict['resolution'],
                                         attributes = edge_dict['attributes'])
        
    return e

def _read_wire(wire_dict: dict) -> topobj.Wire:
    """
    reads the wire of a geomie3d file.
 
    Parameters
    ----------
    wire_dict : dict
        dictionary of the wire from the file. Refer to the wire object for the schema
    
    Returns
    -------
    wire : topobj.Wire
        the wire object.
    """
    edged_list = wire_dict['edge_list']
    edge_list = [_read_edge(ed) for ed in edged_list]
    w = create.wire_frm_edges(edge_list, attributes = wire_dict['attributes'])
        
    return w

def _read_face(face_dict: dict) -> topobj.Face:
    """
    reads the face of a geomie3d file.
 
    Parameters
    ----------
    face_dict : dict
        dictionary of the face from the file. Refer to the face object for the schema
    
    Returns
    -------
    face : topobj.Face
        the face object.
    """
    vlist = create.vertex_list(face_dict['vertex_list'])
    hole_vlist2d = [create.vertex_list(h) for h in face_dict['hole_vertex_list'] ]
    f = create.polygon_face_frm_verts(vlist, hole_vertex_list=hole_vlist2d, attributes = face_dict['attributes'])
    return f

def _read_shell(shell_dict: dict) -> topobj.Shell:
    """
    reads the shell of a geomie3d file.
 
    Parameters
    ----------
    shell_dict : dict
        dictionary of the shell from the file. Refer to the shell object for the schema
    
    Returns
    -------
    shell : topobj.Shell
        the shell object.
    """
    faced_list = shell_dict['face_list']
    face_list = [_read_face(faced) for faced in faced_list]
    s = create.shell_frm_faces(face_list, attributes = shell_dict['attributes'])
    return s

def _read_solid(solid_dict: dict) -> topobj.Solid:
    """
    reads the shell of a geomie3d file.
 
    Parameters
    ----------
    solid_dict : dict
        dictionary of the solid from the file. Refer to the solid object for the schema
    
    Returns
    -------
    solid : topobj.Solid
        the solid object.
    """
    shell = _read_shell(solid_dict['shell'])
    s = create.solid_frm_shell(shell, attributes = solid_dict['attributes'])
    return s

def _read_composite(composite_dict: dict) -> topobj.Composite:
    """
    reads the shell of a geomie3d file.
 
    Parameters
    ----------
    solid_dict : dict
        dictionary of the solid from the file. Refer to the solid object for the schema
    
    Returns
    -------
    composite : topobj.Composite
        the composite object.
    """
    vertexds = composite_dict['vertex_list']
    vertex_list = [_read_vertex(vd) for vd in vertexds]
    
    edgeds = composite_dict['edge_list']
    edge_list = [_read_edge(ed) for ed in edgeds]
    
    wireds = composite_dict['wire_list']
    wire_list = [_read_wire(wd) for wd in wireds]
    
    faceds = composite_dict['face_list']
    face_list = [_read_face(fd) for fd in faceds]
    
    shellds = composite_dict['shell_list']
    shell_list = [_read_shell(sd) for sd in shellds]
    
    solidds = composite_dict['solid_list']
    solid_list = [_read_solid(sod) for sod in solidds]
    
    cmpds = composite_dict['composite_list']
    composite_list = [_read_composite(cmpd) for cmpd in cmpds]
    
    topo_list = []
    topo_list.extend(vertex_list)
    topo_list.extend(edge_list)
    topo_list.extend(wire_list)
    topo_list.extend(face_list)
    topo_list.extend(shell_list)
    topo_list.extend(solid_list)
    topo_list.extend(composite_list)
    
    c = create.composite(topo_list,attributes = composite_dict['attributes'])
    return c

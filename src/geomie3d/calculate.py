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
from numpy.linalg import matrix_rank

from . import get
from . import modify
from . import geom
from . import create
from . import topobj
from . import utility
from . import settings

def a_connected_path_from_edges(edge_list: list[topobj.Edge], indx: bool = False) -> dict:
    """
    loop through the edges and find a path if the edges are connected. If there are multiple connected paths or branches in the edge list, branches are ignored. Only a single path is returned.
 
    Parameters
    ----------
    edge_list : list[topobj.Edge]
        the list of edge object to check if they are connected.
    
    indx : bool, optional
        return the index instead of the edge object.

    Returns
    -------
    result : dict
        A dictionary 
        - connected: list[topobj.Edge]
        - loose: list[topobj.Edge]
        - is_path_closed: bool, indicate whether if the connected path is close or open. True is closed, False is opened.
        - branches: list[list[int]], a list of list of edge index indicating at which edge a branch occur.
        - If indx==True, return the indices instead.
    """
    nedges = len(edge_list)
    all_indxs = range(nedges)
    connected_indx_all = []
    branches = []
    prev_cnt_ls = []
    cnt = 0
    while nedges > 0:
        if cnt not in prev_cnt_ls: # if 
            edge_list_dup = edge_list[:]
            del edge_list_dup[cnt]
            res_dict = find_edges_connected2this_edge(edge_list[cnt], edge_list_dup, mode='end_start', indx=True)
            connected_indxs = res_dict['connected']
            n_connections = len(connected_indxs)
            if n_connections != 0:
                if n_connections > 1:
                    branch_indxs = connected_indxs[1:]
                    branch_indxs_calibrated = np.where(branch_indxs >= cnt, branch_indxs+1, branch_indxs)
                    branch = np.insert(branch_indxs_calibrated, 0, cnt).tolist()
                    branches.append(branch)

                if cnt not in connected_indx_all:
                    connected_indx_all.append(cnt)
                first_connection = connected_indxs[0]
                if first_connection >= cnt:
                    first_connection += 1
                connected_indx_all.append(first_connection)
                prev_cnt_ls.append(cnt)
                cnt = first_connection
            else:
                prev_cnt_ls.append(cnt)
                not_in_cnt = utility.find_xs_not_in_ys(all_indxs, prev_cnt_ls).tolist()
                cnt = not_in_cnt[0]

        nedges = nedges-1
    
    is_path_closed = False
    if connected_indx_all[0] == connected_indx_all[-1]:
        is_path_closed = True
        connected_indx_all = connected_indx_all[0:-1]

    loose_indxs = utility.find_xs_not_in_ys(all_indxs, connected_indx_all).tolist()

    if indx == True:
        res = {'connected': connected_indx_all, 'loose': loose_indxs, 'is_path_closed': is_path_closed, 'branches': branches}
    else:
        connected_edges = np.take(edge_list, connected_indx_all).tolist()
        loose_edges = np.take(edge_list, loose_indxs).tolist()
        res = {'connected': connected_edges, 'loose': loose_edges, 'is_path_closed': is_path_closed, 'branches': branches}
    return res

def angle_btw_2vectors(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """
    This function calculate the angle between two vectors.
 
    Parameters
    ----------    
    vector1 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)], array defining the vector(s).
        
    vector2 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)], array defining the vector(s).
 
    Returns
    -------
    angle : float/list[float]
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

def are_bboxes1_in_bboxes2(bboxes1: list[utility.Bbox], bboxes2: list[utility.Bbox], zdim: bool = True) -> np.ndarray:
    """
    This function calculates if bbox1 is contain inside bbox2.
    
    Parameters
    ----------
    bboxes1 : list[utility.Bbox]
        the bboxes to check

    bboxes2 : list[utility.Bbox]
        the bboxes to check    
    
    zdim : bool, optional
        If True will check the z-dimension. Default = True

    Returns
    -------
    are_related : np.ndarray
        An array of True or False indicating the relationship between the two list of bboxes. True = bboxes are in, False = bboxes are not.
    """
    nbboxes = len(bboxes1)
    bbox_xyzs = create.xyzs_frm_bboxes(bboxes1)
    bboxes_shape = bbox_xyzs.shape
    bbox_xyzs1 = np.reshape(bbox_xyzs, (bboxes_shape[0]*bboxes_shape[1], bboxes_shape[2]))
    indices = match_xyzs_2_bboxes(bbox_xyzs1, bboxes2, zdim = zdim)
    if indices.size == 0:
        are_contained = np.array([False])
        are_contained = np.repeat(are_contained, nbboxes)
        return are_contained
    else:
        xyz_indices = indices[0]
        bboxes1_indices = xyz_indices/8
        bboxes1_indices = np.floor(bboxes1_indices)
        bboxes2_indices = indices[1]
        same_indx = np.where(bboxes1_indices == bboxes2_indices)[0]
        same = np.take(bboxes1_indices, same_indx, axis=0)
        are_contained = []
        for i in range(nbboxes):
            pts_in_bboxes2 = np.where(same == i)[0]
            npts = len(pts_in_bboxes2)
            if npts == 8:
                are_contained.append(True)
            else:
                are_contained.append(False)
        return np.array(are_contained)

def are_bboxes1_related2_bboxes2(bboxes1: list[utility.Bbox], bboxes2: list[utility.Bbox], zdim: bool = True) -> np.ndarray:
    """
    This function calculates if bbox1 is related to bbox2. Related include any form of intersection or touching.
    
    Parameters
    ----------
    bboxes1 : list[utility.Bbox]
        the bboxes to check

    bboxes2 : list[utility.Bbox]
        the bboxes to check    
    
    zdim : bool, optional
        If True will check the z-dimension. Default = True

    Returns
    -------
    are_related : np.ndarray
        An array of True or False indicating the relationship between the two list of bboxes. True = bboxes are related, False = bboxes are not related.
    """
    def are_bboxes_related(bboxes1, bboxes2, nbboxes, zdim):
        bbox_xyzs = create.xyzs_frm_bboxes(bboxes1)
        bboxes_shape = bbox_xyzs.shape
        bbox_xyzs1 = np.reshape(bbox_xyzs, (bboxes_shape[0]*bboxes_shape[1], bboxes_shape[2]))
        indices = match_xyzs_2_bboxes(bbox_xyzs1, bboxes2, zdim = zdim)
        are_related = np.array([False])
        are_related = np.repeat(are_related, nbboxes)
        if indices.size == 0:
            return are_related 
        else:
            xyz_indices = indices[0]
            bboxes1_indices = xyz_indices/8
            bboxes1_indices = np.floor(bboxes1_indices)
            bboxes2_indices = indices[1]
            bboxes_indices = np.array([bboxes1_indices, bboxes2_indices])
            same_indx = np.where(bboxes1_indices == bboxes2_indices)[0]
            bboxes_indicesT = bboxes_indices.T
            same = np.take(bboxes_indicesT, same_indx, axis=0)
            same = np.unique(same, axis=0)
            sameT = same.T[0].astype(int)
            np.put(are_related, sameT, True)
            return are_related

    nbboxes = len(bboxes1)
    are_related1 = are_bboxes_related(bboxes1, bboxes2, nbboxes, zdim)
    are_related2 = are_bboxes_related(bboxes2, bboxes1, nbboxes, zdim)
    are_related_res = np.logical_or(are_related1, are_related2)
    return are_related_res

def are_polygon_faces_convex(faces: list[topobj.Face], ndecimals: int = None) -> np.ndarray:
    """
    check if the polygon faces are convex 
 
    Parameters
    ----------
    faces : list[topobj.Face]
        faces to check whether they are convex.

    ndecimals: int, optional
        the number of decimals to round off to compare if points are the same.
 
    Returns
    -------
    are_face_convex: np.ndarray
        np.ndarray[shape(number of faces)], True or False if the face is convex.
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS

    are_convex = []
    for f in faces:
        verts = get.vertices_frm_face(f)
        xyzs = np.array([v.point.xyz for v in verts])
        xyzs_roll = np.roll(xyzs, -1, axis=0)
        vects = xyzs_roll - xyzs
        vects_roll = np.roll(vects, -1, axis=0)
        cres = cross_product(vects, vects_roll)
        cres_n = normalise_vectors(cres)
        cres_n = np.round(cres_n, decimals=ndecimals)
        uniq = np.unique(cres_n, axis=0)
        if len(uniq) == 1:
            are_convex.append(True)
        else:
            are_convex.append(False)
    return are_convex

def are_verts_in_polygons(verts: list[list[topobj.Vertex]], polygons: list[topobj.Face], atol: float = None, rtol: float = None) -> list[list[bool]]:
    """
    - calculate if verts is in polygons. If the point lies on the edges of the polygons it is counted as inside the polygon.
    - implemented based on https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
 
    Parameters
    ----------
    verts : list[list[topobj.Vertex]]
        the sets of verts to check if they are inside the set of polygons.
    
    polygons: list[topobj.Face]
        the set of polygons.

    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.
 
    Returns
    -------
    list[list[bool]]
        list[shape(number of sets of points, number of points in the set)], True or False if the point is in the polygon.
    """
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    # TODO: ACCOUNT FOR POLYGON HOLES
    xyz_2dlist = []
    for vset in verts:
        xyzs = [v.point.xyz.tolist() for v in vset]
        # print(xyzs)
        xyz_2dlist.append(xyzs)

    polyxyzs_list = []
    for polygon in polygons:
        # polyvs = get.vertices_frm_face(polygon)
        polyvs = get.bdry_vertices_frm_face(polygon)
        polyxyzs = [v.point.xyz.tolist() for v in polyvs]
        polyxyzs_list.append(polyxyzs)

    # print(xyz_2dlist)
    # print(polyxyzs_list)
    are_xyzs_in_poly = are_xyzs_in_polyxyzs(xyz_2dlist, polyxyzs_list, atol=atol, rtol=rtol)
    return are_xyzs_in_poly

def are_xyzs_in_polyxyzs(xyz_2dlist: list[list[list[float]]], polyxyzs: list[list[list[float]]], atol: float = None, rtol: float = None) -> list[list[bool]]:
    """
    - calculate if point xyzs is in polygon xyzs. If the point lies on the edges of the polygons it is counted as inside the polygon.
    - implemented based on https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule
 
    Parameters
    ----------
    xyz_2dlist : np.ndarray
        np.ndarray[shape(number of sets of points, number of points, 3)].
    
    polyxyzs : list[list[list[float]]]
        list[shape(number of polygons, number of points in polygon, 3)].

    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.

    Returns
    -------
    is_xyzs_in_poly: list[list[bool]]
        list[shape(number of sets of points, number of points in the set)], True or False if the point is in the polygon.
    """
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    nsets = len(xyz_2dlist)
    npolys = len(polyxyzs)
    if nsets != npolys:
        print('ERROR NUMBER OF SETS OF POINTS NOT EQUAL TO NUMBER OF POLYGONS')
        return None
    
    each_set_cnt = []
    for setx in xyz_2dlist:
        each_set_cnt.append(len(setx))

    # process the points
    xyz_2dlist_hmg = modify.hmgnz_xyz2dlist(xyz_2dlist)
    xyz_shape = np.shape(xyz_2dlist_hmg)
    xyzs_flat = np.reshape(xyz_2dlist_hmg, (xyz_shape[0]*xyz_shape[1], xyz_shape[2]))

    # process the polyxyzs
    polyxyzs_hmg = modify.hmgnz_xyz2dlist(polyxyzs)
    polyxyzs_hmg = np.repeat(polyxyzs_hmg, xyz_shape[1], axis=0)

    # append the points to the corresponding polys
    xyzs2 = np.reshape(xyzs_flat, (len(xyzs_flat), 1, 3))
    allxyzs = np.append(xyzs2, polyxyzs_hmg, axis=1)
    # check if the points and polys are coplanar
    coplanar_xyzs = allxyzs[:, 0:4]
    coplanar_xyzs = np.roll(coplanar_xyzs, -1, axis=1)
    isnan = np.isnan(coplanar_xyzs)
    isnan_shp = np.shape(isnan)
    isnan = np.reshape(isnan, (isnan_shp[0], isnan_shp[1]*isnan_shp[2]))
    isnan = isnan.T[0]
    is_coplanar = np.logical_not(isnan)
    is_nt_nan_id = np.where(is_coplanar)[0]
    coplanar_xyzs = np.take(coplanar_xyzs ,is_nt_nan_id, axis=0)
    is_coplanar2 = is_coplanar_xyzs(coplanar_xyzs, atol = atol, rtol = rtol)
    np.put(is_coplanar, is_nt_nan_id, is_coplanar2)
    is_coplanar_id = np.where(is_coplanar)[0]
    # if all are not coplanar should return all false and stop the calculation
    if is_coplanar_id.size == 0:
        # print('IS NOT COPLANAR')
        res2 = []
        for each_cnt in range(nsets):
            res1 = []
            for _ in range(each_set_cnt[each_cnt]):
                res1.append(False)
            res2.append(res1)
        return res2
    else:        
        # get the mid point of one of the edges of the polygon
        poly_first_edges = allxyzs[:, 1:3]
        poly_first_edges = np.take(poly_first_edges, is_coplanar_id, axis = 0)
        poly_edge_mid = xyzs_mean(poly_first_edges)
        # print('poly_edge', poly_first_edges)
        points2test = allxyzs[:, 0]
        points2test = np.take(points2test, is_coplanar_id, axis=0)

        # extend the point on the bbox
        pcenters = np.take(allxyzs[:, 1:], is_coplanar_id, axis = 0)
        pcenters = xyzs_mean(pcenters)
        dirx = poly_edge_mid - pcenters
        dirx = normalise_vectors(dirx)
        mv_poly_edge_mid = move_xyzs(poly_edge_mid, dirx, np.repeat([10], len(dirx)))
        # append the point to the bbox to create the edge
        p2t_shp = np.shape(points2test)
        points2test = np.reshape(points2test, [p2t_shp[0], 1, p2t_shp[1]])
        mv_poly_edge_mid = np.reshape(mv_poly_edge_mid, [p2t_shp[0], 1, p2t_shp[1]])
        test_edges = np.append(points2test, mv_poly_edge_mid, axis=1)

        # get all the corresponding polygons
        poly2test = allxyzs[:, 1:]
        poly2test = np.take(poly2test, is_coplanar_id, axis=0)
        p2t_edges, is_nt_nan_id2_intx, intx_res, poly2t_shp, is_nt_nan_id2 = _rmv_nan_frm_hmgn_polys_edges_xyzs(poly2test)
        
        # reshape the test edges to match the poly xyzs shape to perform the line-line intersection
        tedges_rep = np.repeat(test_edges, int(poly2t_shp[1]/2), axis=0)
        tedges_shp = np.shape(tedges_rep)
        tedges_rep_flat = np.reshape(tedges_rep, [tedges_shp[0]*tedges_shp[1], tedges_shp[2]])
        tedges_rep_flat_nt_nan = np.take(tedges_rep_flat, is_nt_nan_id2, axis=0)
        tedges_rep_flat_nt_nan_shp = np.shape(tedges_rep_flat_nt_nan)
        tedges_rep_ready = np.reshape(tedges_rep_flat_nt_nan, [int(tedges_rep_flat_nt_nan_shp[0]/2), 2, tedges_rep_flat_nt_nan_shp[1]])

        intxs = linexyzs_intersect(tedges_rep_ready, p2t_edges, atol=atol, rtol=rtol)
        # print('intxs', intxs)

        np.put_along_axis(intx_res, is_nt_nan_id2_intx, intxs, axis=0)
        intx_res = np.reshape(intx_res, [int((poly2t_shp[0] * poly2t_shp[1]/2)/(poly2t_shp[1]/2)), int(poly2t_shp[1]/2), 3])
        # check if the number of intx is odd or even number
        for cnt,intx in enumerate(intx_res):
            is_nan_intx = np.isnan(intx.T[0])
            is_nt_nan_intx = np.logical_not(is_nan_intx)
            is_nt_nan_intx_id = np.where(is_nt_nan_intx)[0]
            intx_nt_nan = np.take(intx, is_nt_nan_intx_id, axis=0)
            intx_uniq = np.unique(intx_nt_nan, axis=0)
            
            nintx = len(intx_uniq)
            if nintx%2 == 0:
                in_poly = False
            else:
                in_poly = True
            np.put(is_coplanar, is_coplanar_id[cnt], in_poly)

        # sort the results into the right dimension
        is_coplanar_2d = np.reshape(is_coplanar, [xyz_shape[0], xyz_shape[1]])
        res2 = []
        for each_cnt, a_coplanar in enumerate(is_coplanar_2d):
            res1 = []
            for i in range(each_set_cnt[each_cnt]):
                res1.append(a_coplanar[i])
            res2.append(res1)

        return res2

def bboxes_centre(bboxes: list[utility.Bbox]) -> np.ndarray:
    """
    This function returns the centre point of the bbox .
    
    Parameters
    ----------
    bbox : list[utility.Bbox]
        list[utility.Bbox] to calculate the center
       
    Returns
    -------
    xyzs : np.ndarray
        np.ndarray(shape(number of bboxes, 3)) array defining the points.
    """
    bbox_arrs = np.array([bbox.bbox_arr for bbox in bboxes])
    bbox_arrsT = bbox_arrs.T
    mnxs = bbox_arrsT[0]
    mnys = bbox_arrsT[1]
    mnzs = bbox_arrsT[2]
    mxxs = bbox_arrsT[3]
    mxys = bbox_arrsT[4]
    mxzs = bbox_arrsT[5]

    midxs = mnxs + ((mxxs - mnxs)/2)
    midys = mnys + ((mxys - mnys)/2)
    midzs = mnzs + ((mxzs - mnzs)/2)
    center_pts = np.vstack([midxs, midys, midzs]).T

    return center_pts

def bbox_frm_bboxes(bbox_list: list[utility.Bbox]) -> utility.Bbox:
    """
    This function recalculate the bbox from a list of bboxes.
    
    Parameters
    ----------
    bbox_list : list[utility.Bbox]
        A list of bboxes    
    
    Returns
    -------
    bbox : utility.Bbox
        the calculated bbox object
    """
    bbox_arr_ls = [bbox.bbox_arr.tolist() for bbox in bbox_list]
    zip_box = list(zip(*bbox_arr_ls))
    mnx = min(zip_box[0])
    mny = min(zip_box[1])
    mnz = min(zip_box[2])
    
    mxx = max(zip_box[3])
    mxy = max(zip_box[4])
    mxz = max(zip_box[5])
    
    res_bbox = utility.Bbox([mnx,mny,mnz,mxx,mxy,mxz])
    return res_bbox

def bbox_frm_topo(topo: topobj.Topology | list[topobj.Topology]) -> utility.Bbox | list[utility.Bbox]:
    """
    calculate the bbox from a topology object.
    
    Parameters
    ----------
    topo : topobj.Topology/list[topobj.Topology]
        topobj.Topology or list[topobj.Topology], the topology to analyse.

    Returns
    -------
    bbox : utility.Bbox/list[utility.Bbox]
        bbox object
    """
    def ext_xyzs(t):
        verts = get.topo_explorer(t, topobj.TopoType.VERTEX)
        xyzs = np.array([v.point.xyz for v in verts])
        return xyzs
    
    if isinstance(topo, list):
        xyz_ls = []
        for t in topo:
            xyzs = ext_xyzs(t)
            xyz_ls.append(xyzs)
        bbox = bbox_frm_xyzs(xyz_ls)
    else:    
        xyzs = ext_xyzs(topo)
        bbox = bbox_frm_xyzs(xyzs)
    return bbox

def bbox_frm_xyzs(xyzs: list) -> utility.Bbox | list[utility.Bbox]:
    """
    This function returns the bbox of the xyzs.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)] or np.ndarray[shape(number of sets of points, number of points, 3)] array defining the points.

    Returns
    -------
    bbox : utility.Bbox/list[utility.Bbox]
        bbox object or list of bbox object
    """
    flat = list(chain.from_iterable(xyzs))
    flat_shp = np.shape(flat)
    nshape = len(flat_shp)
    if nshape == 1: #only 1 set of points
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
        
        bbox = utility.Bbox([mnx,mny,mnz,mxx,mxy,mxz])
        return bbox
    
    if nshape == 2:
        xyz_2darr = modify.hmgnz_xyz2dlist(xyzs)
        shp = np.shape(xyz_2darr)
        xyz_2darr_flat = np.reshape(xyz_2darr, (shp[0]*shp[1], shp[2]))
        xyzst = xyz_2darr_flat.T
        xs = xyzst[0]
        ys = xyzst[1]
        zs = xyzst[2]

        xs = np.reshape(xs, (shp[0], shp[1]))
        ys = np.reshape(ys, (shp[0], shp[1]))
        zs = np.reshape(zs, (shp[0], shp[1]))

        mnx = np.nanmin(xs, axis=1)
        mxx = np.nanmax(xs, axis=1)

        mny = np.nanmin(ys, axis=1)
        mxy = np.nanmax(ys, axis=1)

        mnz = np.nanmin(zs, axis=1)
        mxz = np.nanmax(zs, axis=1)

        bbox_xyzs = np.vstack([mnx,mny,mnz,mxx,mxy,mxz])
        bbox_xyzsT = bbox_xyzs.T
        bboxes = [utility.Bbox(bbox_xyzs) for bbox_xyzs in bbox_xyzsT]
        return bboxes
    
def cross_product(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    This function cross product two vectors.It is a wrap of the numpy cross function
 
    Parameters
    ----------    
    vector1 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)], array defining the vector(s).
        
    vector2 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)], array defining the vector(s).
 
    Returns
    -------
    cross_product : np.ndarray
        np.ndarray of the cross product
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
     
    cross_product = np.cross(vector1, vector2)
    return cross_product

def cs2cs_matrice(orig_cs: utility.CoordinateSystem, dest_cs: utility.CoordinateSystem) -> np.ndarray:
    """
    This function calculate a 4x4 matrice for transforming one coordinate to another coordinate system.
    - https://gamedev.stackexchange.com/questions/203073/how-to-convert-a-4x4-matrix-transformation-to-another-coordinate-system
 
    Parameters
    ----------    
    orig_cs : utility.CoordinateSystem
        the original coordinate system.
    
    dest_cs : utility.CoordinateSystem
        the coordinate system to transform to.
 
    Returns
    -------
    matrice : np.ndarray
        4x4 matrice.
    """
    #first move the orig cs to [0,0,0] origin and xdir = [1,0,0], ydir =[0,1,0] and zdir = [0,0,1]
    orig = np.array([0,0,0])
    orig1 = orig_cs.origin
    orig2 = dest_cs.origin
    trsl = orig - orig1
    trsl_mat = translate_matrice(trsl[0],trsl[1],trsl[2])
    xdir = orig_cs.x_dir
    ydir = orig_cs.y_dir
    zdir = cross_product(xdir, ydir)
    trsf_mat = [[xdir[0], ydir[0], zdir[0], 0], 
                [xdir[1], ydir[1], zdir[1], 0],
                [xdir[2], ydir[2], zdir[2], 0],
                [0, 0, 0, 1]]
    trsf_mat = inverse_matrice(trsf_mat)

    #then move to the dest cs
    xdir2 = dest_cs.x_dir
    ydir2 = dest_cs.y_dir
    zdir2 = cross_product(xdir2, ydir2)
    trsf_mat2 = [[xdir2[0], ydir2[0], zdir2[0], 0], 
                 [xdir2[1], ydir2[1], zdir2[1], 0],
                 [xdir2[2], ydir2[2], zdir2[2], 0],
                 [0, 0, 0, 1]]
    trsl2 = orig2 - orig
    trsl_mat2 = translate_matrice(trsl2[0],trsl2[1],trsl2[2])
    cs2cs_mat = trsl_mat2@trsf_mat2@trsf_mat@trsl_mat
    return cs2cs_mat

def dist_btw_xyzs(xyzs1: np.ndarray, xyzs2: np.ndarray) -> np.ndarray:
    """
    This function calculates the distance between two xyzs. 
 
    Parameters
    ----------
    xyzs1 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of points, 3)].
    
    xyzs2 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of points, 3)].
    
    Returns
    -------
    distance : np.ndarray
        np.ndarray[shape(number of points)]. distances between the points.
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

def dist_pointxyzs2linexyzs(pointxyzs: np.ndarray, linexyzs: np.ndarray, int_pts: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Find the distance between the points and the lines. Based on this post https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    Parameters
    ----------
    pointxyzs : np.ndarray
        np.ndarray[shape(number of points, 3)], array of points [point1, point2, ...]. Each point is define as [x,y,z].
    
    linexyzs : np.ndarray
        np.ndarray[shape(number of lines, 2, 3)], array of edges [edge1, edge2, ...]. Each edge is define as [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]. 
    
    int_pts: bool, optional
        if true will return the closest point on the line to the point. default == False
        
    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        - dist: np.ndarray[shape(1)], distances.
        - closest_pts : np.ndarray[shape(number of points, 3)], optional,  the closest points on the line to the point.
    """
    if type(pointxyzs) != np.ndarray:
        pointxyzs = np.array(pointxyzs)
        
    if type(linexyzs) != np.ndarray:
        linexyzs = np.array(linexyzs)
    
    shape = np.shape(linexyzs)
    xyzs = np.reshape(linexyzs, (shape[0]*shape[1], shape[2]))
    xyzs1 = xyzs[0::2]
    xyzs2 = xyzs[1::2]
    
    a = xyzs1 - pointxyzs
    b = xyzs2 - xyzs1
    c = dot_product(a, b)
    
    dist = abs(dist_btw_xyzs(xyzs2, xyzs1))
    distsq = np.square(dist)
    ts = c/distsq
    ts = ts*-1
    ts = np.where(ts < 0, 0, ts)
    ts = np.where(ts > 1, 1, ts)
    int_xyzs = linexyzs_from_t(ts, linexyzs)
    dists = dist_btw_xyzs(pointxyzs, int_xyzs)
    if int_pts == False:
        return dists
    else:
        return dists, int_xyzs

def dist_pointxyzs2polyxyzs(xyzs: np.ndarray, polyxyzs: np.ndarray, nrmlxyzs: np.ndarray, int_pts: bool = False, 
                            atol: float = None, rtol: float = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Find the distance between the points and the polygons.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)], array of points [point1, point2, ...]. Each point is define as [x,y,z].
    
    polyxyzs : np.ndarray
        np.ndarray[shape(number of polygons, number of points in polygons, 3)]. 
    
    nrmlxyzs : np.ndarray
        np.ndarray[shape(number of polygons, 3)], normal of each polygon.

    int_pts: bool, optional
        if true will return the closest point on the polygon to the point. default == False
    
    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.
        
    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        - dist: np.ndarray[shape(number of points)], distances.
        - closest_pts : np.ndarray[shape(number of points, 3)], optional, the closest points on the line to the point.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)

    if type(nrmlxyzs) != np.ndarray:
        nrmlxyzs = np.array(nrmlxyzs)

    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    nxyzs = len(xyzs)
    npolys = len(polyxyzs)
    if nxyzs != npolys:
        print('ERROR NUMBER OF POINTS NOT EQUAL TO NUMBER OF POLYGONS')
        return None
    
    # process the polyxyzs
    polyxyzs_hmg = modify.hmgnz_xyz2dlist(polyxyzs)
    # project the point onto the plane of the polygon
    abcds = planes_frm_pointxyzs_nrmls(polyxyzs_hmg[:, 0], nrmlxyzs)
    dist2pl, ptxyzs_on_pl = proj_ptxyzs_on_plane(xyzs, abcds)
    # check if the points are in the polygons
    on_pl_shape = np.shape(ptxyzs_on_pl)
    ptxyzs_on_pl_2d = np.reshape(ptxyzs_on_pl, (on_pl_shape[0], 1, on_pl_shape[1]))
    are_in_polys = are_xyzs_in_polyxyzs(ptxyzs_on_pl_2d, polyxyzs, atol=atol, rtol=rtol)
    # coplanar = np.vstack([polyxyzs[0], ptxyzs_on_pl])
    # is_coplanar = is_coplanar_xyzs(coplanar)
    are_in_polys = np.reshape(are_in_polys, (on_pl_shape[0]))
    are_nt_in_polys = np.logical_not(are_in_polys)
    dist2polys = []
    poly_intxs = []
    for cnt, is_nt in enumerate(are_nt_in_polys):
        if is_nt == True:
            # need to do intersection between points and edges
            poly_edges = _polyxyzs2edges(polyxyzs[cnt])
            nedges = len(poly_edges)
            pts2int = np.array([ptxyzs_on_pl[cnt]])
            pts2int_rep = np.repeat(pts2int, nedges, axis=0)
            dist2edges, intxyz2edges = dist_pointxyzs2linexyzs(pts2int_rep, poly_edges, int_pts=True)
            min_dist = np.min(dist2edges)
            min_dist_indx = np.where(dist2edges == min_dist)[0][0]
            intxyz = intxyz2edges[min_dist_indx]
            dist2poly = dist_btw_xyzs(xyzs[cnt], intxyz)
            if int_pts == True:
                poly_intxs.append(intxyz)
        else:
            pts2int = np.array(ptxyzs_on_pl[cnt])
            dist2poly = dist2pl[cnt]
            if int_pts == True:
                poly_intxs.append(pts2int)
            
        dist2polys.append(dist2poly)
    
    dist2polys = np.array(dist2polys)
    if int_pts == True:
        return dist2polys, np.array(poly_intxs)
    else:
        return dist2polys

def dist_vertex2line_edge(vertex_list: list[topobj.Vertex], edge_list: list[topobj.Edge], int_pts: bool = False) -> np.ndarray:
    """
    Find the distance between the vertices to the edges. The edges cannot have curve that has more than 2 points. Only work with lines.
    
    Parameters
    ----------
    vertex_list : list[topobj.Vertex]
        list of vertex.
    
    edge_list : list[topobj.Edge]
        list of edges. The edges can only have line as geometry. That is polygon curve that is define by only two vertices.
    
    int_pts: bool, optional
        if true will return the closest point on the line to the point. default == False
        
    Returns
    -------
    distances : np.ndarray
        array of all the distances.
    
    closest_verts : list[topobj.Vertex], optional
        the closest vertices on the line to the point.
    """
    pointxyzs = [v.point.xyz for v in vertex_list]
    
    linexyzs = _extract_xyzs_from_lineedge(edge_list)
    #TODO implement for more complex polygon curve can be achieve by breaking each curve into lines
    
    if int_pts == False:
        dists = dist_pointxyzs2linexyzs(pointxyzs, linexyzs, int_pts = False)
        return dists
    else:
        dists, intxs = dist_pointxyzs2linexyzs(pointxyzs, linexyzs, int_pts = True)
        vlist = create.vertex_list(intxs)
        return dists, vlist

def dist_verts2polyfaces(vertex_list: list[topobj.Vertex], polyfaces: list[topobj.Face], int_pts: bool = False,
                         atol: float = None, rtol: float = None) -> np.ndarray | tuple[np.ndarray, list[topobj.Vertex]]:
    """
    Find the distance between the verts and the polygons. Does not accound for the hole in the polygons. Only boundary is used.
    
    Parameters
    ----------
    vertex_list: list[topobj.Vertex]
        list of vertexs.
    
    polyfaces: list[topobj.Face]
        list of polygon faces to measure distance to.

    int_pts: bool, optional
        if true will return the closest point on the polygon to the point. default == False
    
    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.
        
    Returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        - dist: np.ndarray[shape(number of points)], distances.
        - closest_pts : list[topobj.Vertex].
    """
    # TODO: account for holes in polygons
    if len(vertex_list) != len(polyfaces):
        print('ERROR NUMBER OF VERTEX DOES NOT MATCH POLYGON FACES')
        return None
    
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    pointxyzs = [v.point.xyz for v in vertex_list]
    polyxyzs = []
    nrmls = []
    for f in polyfaces:
        fvs = get.bdry_vertices_frm_face(f)
        nrml = get.face_normal(f)
        nrmls.append(nrml)
        xyzs = [v.point.xyz.tolist() for v in fvs]
        polyxyzs.append(xyzs)

    if int_pts == False:
        dists = dist_pointxyzs2polyxyzs(pointxyzs, polyxyzs, nrmls, int_pts=False, atol=atol, rtol=rtol)
        return dists
    else:
        dists, intxs = dist_pointxyzs2polyxyzs(pointxyzs, polyxyzs, nrmls, int_pts=True, atol=atol, rtol=rtol)
        vlist = create.vertex_list(intxs)
        return dists, vlist

def dot_product(vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
    """
    This function cross product two vectors. Wrap of numpy dot function
 
    Parameters
    ----------    
    vector1 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)], array defining the vector(s).
        
    vector2 : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)], array defining the vector(s).
 
    Returns
    -------
    dot_product : np.ndarray
        numpy array of the dot product
    """
    if type(vector1) != np.ndarray:
        vector1 = np.array(vector1)
    if type(vector2) != np.ndarray:
        vector2 = np.array(vector2)
    
    shape = np.shape(vector1)
    nshape = len(shape)
    if nshape == 1:
        dot_product = np.dot(vector1, vector2)
        return dot_product
    elif nshape == 2:
        v1T = vector1.T
        v2T = vector2.T
        a = v1T[0]*v2T[0]
        b = v1T[1]*v2T[1]
        c = v1T[2]*v2T[2]
        dot_product = a+b+c
        return dot_product

def face_area(face: topobj.Face) -> float:
    """
    Calculates the area of the face
 
    Parameters
    ----------    
    face : topobj.Face
        the face object.
    
    Returns
    -------
    area : float
        the area of the face.
    """
    def unit_normal(a, b, c):
        x = np.linalg.det([[1,a[1],a[2]],
             [1,b[1],b[2]],
             [1,c[1],c[2]]])
        y = np.linalg.det([[a[0],1,a[2]],
             [b[0],1,b[2]],
             [c[0],1,c[2]]])
        z = np.linalg.det([[a[0],a[1],1],
             [b[0],b[1],1],
             [c[0],c[1],1]])
        magnitude = (x**2 + y**2 + z**2)**.5
        return (x/magnitude, y/magnitude, z/magnitude)
    
    if face.surface_type == geom.SrfType.POLYGON:
        verts = get.vertices_frm_face(face)
        nverts = len(verts)
        if nverts < 3: #not a plane no area
            return 0
        total = [0, 0, 0]
        for i in range(nverts):
            vi1 = verts[i]
            vi1 = vi1.point.xyz
            vi2 = verts[(i+1) % nverts]
            vi2 = vi2.point.xyz
            prod = np.cross(vi1, vi2)
            total[0] += prod[0]
            total[1] += prod[1]
            total[2] += prod[2]
            
        result = np.dot(total, unit_normal(verts[0].point.xyz, verts[1].point.xyz, verts[2].point.xyz))
        return abs(result/2)
    
    elif face.surface_type == geom.SrfType.BSPLINE:
        print('Area of face with Bspline surface not yet implemented, convert bspline to polygon surface')

def face_midxyz(face: topobj.Face) -> np.ndarray:
    """
    Calculates the midpt of the face
 
    Parameters
    ----------    
    face : topobj.Face
        the face object.
    
    Returns
    -------
    midxyz : np.ndarray
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

def find_edges_connected2this_edge(this_edge: topobj.Edge, edge_list: list[topobj.Edge], mode: str = 'end_start', indx: bool = False, atol: float = None, 
                                   rtol: float = None) -> dict:
    """
    find the edges in the edge_list that is connected to this_edge. The algorithm use the end point of this_edge and check it with the start point of the edge_list, directionality of the edges matter.
 
    Parameters
    ----------
    this_edge: topobj.Edge
        the edge object.

    edge_list : list[topobj.Edge]
        the list of edge object to check if they are connected to this_edge.

    mode : str, optional
        the mode to determine the way edges are connected. Default to end_start
        - end_start: the end point of this_edge to the start point of the edge_list
        - start_end: the start point of this_edge to the end point of the edge_list
        - end_end: the end point of this_edge to the end point of the edge_list
        - start_start: the start point of this_edge to the start point of the edge_list 

    indx : bool, optional
        return the index instead of the actual edge object.

    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.
 
    Returns
    -------
    connected_edges : dict
        A dictionary {'connected': list[topobj.Edge], 'loose': list[topobj.Edge]}. If indx==True, return the indices instead.
    """
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    if mode == 'end_start':
        this_xyz = this_edge.end_vertex.point.xyz
        xyz_ls = np.array([e.start_vertex.point.xyz for e in edge_list])
    elif mode == 'start_end':
        this_xyz = this_edge.start_vertex.point.xyz
        xyz_ls = np.array([e.end_vertex.point.xyz for e in edge_list])
    elif mode == 'end_end':
        this_xyz = this_edge.end_vertex.point.xyz
        xyz_ls = np.array([e.end_vertex_vertex.point.xyz for e in edge_list])
    elif mode == 'start_start':
        this_xyz = this_edge.start_vertex.point.xyz
        xyz_ls = np.array([e.start_vertex.point.xyz for e in edge_list])
    
    all_indxs = range(len(xyz_ls))    
    connected_indxs = find_these_xyzs_in_xyzs([[this_xyz]], [xyz_ls], atol=atol, rtol=rtol)[1]
    loose_indxs = utility.find_xs_not_in_ys(all_indxs, connected_indxs)
    
    if indx == True:
        result = {'connected': connected_indxs, 'loose': loose_indxs}
    else:
        connected_edges = np.take(edge_list, connected_indxs).tolist()
        loose_edges = np.take(edge_list, loose_indxs).tolist()
        result = {'connected': connected_edges, 'loose': loose_edges}
    return result

def find_faces_outline(face_list: list[topobj.Face], ndecimals: int = None) -> tuple[list[topobj.Edge], list[topobj.Edge]]:
    """
    Find non-duplicated edges from a list of faces. Can be used to find the outline of a triangulated surface.
    
    Parameters
    ----------
    face_list: list[topobj.Face]
        find non-duplicated edges of these faces.

    ndecimals: int, optional
        the number of decimals to round off to compare if points are the same.
    
    Returns
    -------
    non_dup_edges : list[topobj.Edge]
        list of non duplicated edges
    
    dup_edges : list[topobj.Edge]
        list[Shape(Any, Any)] of duplicated edges
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS
    edge_ls = [get.edges_frm_face(f) for f in face_list]
    edge_ls = list(chain(*edge_ls))
    edge_ls = modify.edges2lineedges(edge_ls)
    non_dup_edges, dup_edges = find_non_dup_lineedges(edge_ls, ndecimals=ndecimals)
    return non_dup_edges, dup_edges

def find_non_dup_lineedges(edge_list: list[topobj.Edge], ndecimals: int = None) -> tuple[list[topobj.Edge], list[list[topobj.Edge]]]:
    """
    Find edges that are not duplicated.
    
    Parameters
    ----------
    edge_list: list[topobj.Edge]
        find non duplicated edges from these edges.

    ndecimals: int, optional
        the number of decimals to round off to compare if points are the same.
    
    Returns
    -------
    non_dup_edges : list[topobj.Edge]
        list of non duplicated edges
    
    dup_edges : list[list[topobj.Edge]]
        list[shape(number of duplicates, Any)] of duplicated edges
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS
    edge_vert_list = [get.vertices_frm_edge(e) for e in edge_list]
    edge_xyz_ls = []
    for edge_verts in edge_vert_list:
        edge_xyzs = [v.point.xyz for v in edge_verts]
        edge_xyz_ls.append(edge_xyzs)

    edge_xyz_ls = np.array(edge_xyz_ls)
    non_dup_idx, dup_idx = find_non_dup_lineedges_xyz(edge_xyz_ls, ndecimals=ndecimals)
    non_dup_edges = np.take(edge_list, non_dup_idx, axis=0)
    non_dup_edges = non_dup_edges.tolist()
    dup_edges = np.take(edge_list, dup_idx).tolist()
    return non_dup_edges, dup_edges

def find_non_dup_lineedges_xyz(edge_xyz_list: np.ndarray, index: bool = True, ndecimals: int = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Find edges that are not duplicated.
    
    Parameters
    ----------
    edge_xyz_list: np.ndarray
        np.ndarray(Shape[Any, 2, 3]) find non duplicated edges from these edges.
    
    index: bool, optional
        Default = True, if True return the indices of the edges that are non-duplicated and duplicated. False returns only the xyzs of the non duplicated. 
    
    ndecimals: int, optional
        the number of decimals to round off to compare if points are the same.

    Returns
    -------
    non_dup_edge_xyzs : np.ndarray
        np.ndarray[shape(number of non-duplicated edges, 2, 3)] of non duplicated edges
    
    indices : np.ndarray
        - non_dup_idx = np.ndarray[shape(number of non-dup)]
        - dup_idx = np.ndarray[shape(number of duplicates, any)].
    """
    if type(edge_xyz_list) != np.ndarray:
        np.array(edge_xyz_list)

    if ndecimals is None:
        ndecimals = settings.NDECIMALS

    arr_shape = np.shape(edge_xyz_list)
    if arr_shape[1] != 2:
        raise ValueError('This is not a line edge !! The edge has more than two vertices')
    else:
        xyz_list = np.reshape(edge_xyz_list, (arr_shape[0]*arr_shape[1], 3)) # flatten the list
        xyz_list = np.round(xyz_list, decimals=ndecimals)
        uniq_xyzs = np.unique(xyz_list, axis=0, return_inverse = True) # index the vertices
        edge_idxs = np.reshape(uniq_xyzs[1], (arr_shape[0], arr_shape[1])) # reshape it back to the edge shape
        edge_idxs = np.sort(edge_idxs) # sort the edge indices 
        uniq_edge_idxs = np.unique(edge_idxs, axis=0, return_inverse=True) # compare the edges and identify duplicates
        non_dup_idx, dup_idx = utility.separate_dup_non_dup(uniq_edge_idxs[1]) # separate the dup from non dups
        if index == True:
            return non_dup_idx, dup_idx
        else:
            non_dup = np.take(edge_xyz_list, non_dup_idx)
            return non_dup

def find_these_xyzs_in_xyzs(these_xyzs_2dlist: list[list[list[float]]], xyz_2dlist: list[list[list[float]]], atol: float = None, 
                            rtol: float = None) -> np.ndarray:
    """
    Compare these_xyzs_2dlist to the xyz_2dlist and find all the xyz that is the same as these_xyzs_2dlist. Returns the index of the identical xyz in the xyz_2dlist.
 
    Parameters
    ----------
    these_xyzs_2dlist : list[list[list[float]]]
        list[shape(number of sets of points, number of points, 3)].
    
    xyz_2dlist : list[list[list[float]]]
        list[shape(number of sets of points, number of points, 3)].

    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.
 
    Returns
    -------
    index_list : np.ndarray
        np.ndarray[shape(2, any)], the first index list refers to the index of the set, the second list is the index of the points in that set. 
        So if you do a index_list.T you can get the indices to get the points from the xyz_2dlist
    """
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    nsets = len(these_xyzs_2dlist)
    nsets2 = len(xyz_2dlist)
    if nsets != nsets2:
        print('The number of sets of points do not coincide')
        return False
    
    these_xyzs_arr = modify.hmgnz_xyz2dlist(these_xyzs_2dlist)
    xyz_2darr = modify.hmgnz_xyz2dlist(xyz_2dlist)

    these_shape = np.shape(these_xyzs_arr)
    these_flat = np.reshape(these_xyzs_arr, (these_shape[0]*these_shape[1], these_shape[2]))
    nthese = len(these_flat)
    these_flatT = these_flat.T
    these_flatx = these_flatT[0]
    these_flaty = these_flatT[1]
    these_flatz = these_flatT[2]
    these_flatx = np.reshape(these_flatx, (nthese, 1))
    these_flaty = np.reshape(these_flaty, (nthese, 1))
    these_flatz = np.reshape(these_flatz, (nthese, 1))

    xyz_2darr_rep = np.repeat(xyz_2darr, these_shape[1], axis=0)
    xyz_2darr_rep_shp = np.shape(xyz_2darr_rep)
    xyz_2darr_rep_flat = np.reshape(xyz_2darr_rep, (xyz_2darr_rep_shp[0]*xyz_2darr_rep_shp[1], xyz_2darr_rep_shp[2]))
    xyz_2darr_rep_flatT = xyz_2darr_rep_flat.T
    xyz_ls_x = xyz_2darr_rep_flatT[0]
    xyz_ls_y = xyz_2darr_rep_flatT[1]
    xyz_ls_z = xyz_2darr_rep_flatT[2]
    xyz_ls_x = np.reshape(xyz_ls_x, (xyz_2darr_rep_shp[0], xyz_2darr_rep_shp[1]))
    xyz_ls_y = np.reshape(xyz_ls_y, (xyz_2darr_rep_shp[0], xyz_2darr_rep_shp[1]))
    xyz_ls_z = np.reshape(xyz_ls_z, (xyz_2darr_rep_shp[0], xyz_2darr_rep_shp[1]))

    validx = np.isclose(these_flatx, xyz_ls_x, rtol=rtol, atol=atol) # these_flatx == xyz_ls_x
    validy = np.isclose(these_flaty, xyz_ls_y, rtol=rtol, atol=atol) # these_flaty == xyz_ls_y
    validz = np.isclose(these_flatz, xyz_ls_z, rtol=rtol, atol=atol) # these_flatz == xyz_ls_z

    cond1 = np.logical_and(validx, validy)
    cond2 = np.logical_and(cond1, validz)
    set_indxs, pt_indxs = np.where(cond2)
    set_indxs = np.floor(set_indxs/these_shape[1]).astype(int)
    return np.array([set_indxs, pt_indxs])

def grp_faces_on_nrml(face_list: list[topobj.Face], return_idx: bool = False, ndecimals: int = None) -> tuple[list[list[topobj.Face]], list[topobj.Face]]:
    """
    Group the faces based on the normal of the faces
    
    Parameters
    ----------
    face_list: list[topobj.Face]
        group these faces
    
    return_idx: bool, optional
        only return the indices of the grouped faces. Default to False

    ndecimals: int, optional
        the number of decimals to round the normals of the faces. Default to 5

    Returns
    -------
    res_face : tuple[list[list[topobj.Face]], list[topobj.Face]]
        list[Shape[Any,Any]] of grouped faces. list[Shape[Any]] of faces that do not belong to any group.
        
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS
    # get the normals of each tri face
    nrml_ls = np.array([get.face_normal(f) for f in face_list])
    # if ndecimals != None:
    nrml_ls = np.round(nrml_ls, decimals=ndecimals)
    uniq_nrml = np.unique(nrml_ls, axis=0, return_inverse = True)
    idx = utility.separate_dup_non_dup(uniq_nrml[1])
    non_dup_idx = idx[0]
    dup_idx = idx[1]

    if return_idx == False:
        indv_faces = []
        g_faces = []
        if len(non_dup_idx) !=0:
            indv_faces = np.take(face_list,non_dup_idx).tolist()
        if len(dup_idx) != 0:
            g_faces = [np.take(face_list, idx, axis=0) for idx in dup_idx]
            
        return g_faces, indv_faces
    else:
        return dup_idx, non_dup_idx

def id_bboxes_contain_xyzs(bbox_list: list[utility.Bbox], xyzs: np.ndarray, zdim: bool = True) -> np.ndarray:
    """
    This function returns the indices of the bbox which contains the points.
    
    Parameters
    ----------
    bbox_list : list[utility.Bbox]
        A list of bbox
        
    xyzs : np.ndarray
        array defining the points np.ndarray(shape(number of points, 3)).
            
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    bbox_indices : np.ndarray
        Indices of the boundary that contains the point.
    """
    indices = match_xyzs_2_bboxes(xyzs, bbox_list, zdim = zdim)
    bbox_indices = indices[1]
    bbox_indices = np.unique(bbox_indices)
    
    return bbox_indices

def id_xyzs_in_bboxes(xyzs: np.ndarray, bbox_list: list[utility.Bbox], zdim: bool = False) -> np.ndarray:
    """
    This function returns the indices of the points that are contained by bboxes.
    
    Parameters
    ----------
    xyzs : np.ndarray
        array defining the points np.ndarray(shape(number of points, 3)).
        
    bbox_list : list[utility.Bbox]
       A list of bbox
       
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    pt_indices : np.ndarray
        Indices of the points in the bboxes.
    """
    indices = match_xyzs_2_bboxes(xyzs, bbox_list, zdim = zdim)
    pt_indices = indices[0]
    pt_indices = np.unique(pt_indices)
    
    return pt_indices

def inverse_matrice(matrice: np.ndarray) -> np.ndarray:
    """
    This function inverse the 4x4 matrice.
 
    Parameters
    ----------    
    matrice : np.ndarray
        4x4 matrice to inverse.
    
    Returns
    -------
    matrice : np.ndarray
        inverted 4x4 matrice.
    """
    return np.linalg.inv(matrice)

def is_collinear(vertex_list: list[topobj.Vertex], ndecimals: int = None) -> bool:
    """
    This function checks if the list of points are collinear. 
 
    Parameters
    ----------
    vertex_list : list[topobj.Vertex]
        list[shape(number of vertices)] or list[shape(number of sets of vertices, number of vertices)] array of vertices. It will also work for a 2darray of vertex list
    
    ndecimals: int, optional
        the number of decimals to round off to compare if points are the same.

    Returns
    -------
    True or False : bool
        If True the list of points are collinear.
    """
    if type(vertex_list) != np.ndarray:
        vertex_list = np.array(vertex_list)
    
    if ndecimals is None:
        ndecimals = settings.NDECIMALS

    is_2d = isinstance(vertex_list[0], np.ndarray | list)
    
    if not is_2d:
        xyzs = [v.point.xyz for v in vertex_list]
        
    else:
        xyzs = []
        for verts in vertex_list:
            xyz = [v.point.xyz for v in verts]
            xyzs.append(xyz)
            
    return is_collinear_xyzs(xyzs, ndecimals=ndecimals)

def is_collinear_xyzs(xyzs: np.ndarray, ndecimals: int = None) -> bool:
    """
    This function checks if the list of xyzs are collinear.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray(shape(number of points, 3)) or np.ndarray(shape(number of sets of points, number of points, 3)).

    ndecimals: int, optional
        the number of decimals to round off to compare if points are the same.
        
    Returns
    -------
    True or False : bool
        If True the list of points are coplanar.
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS
    affine_rank = _affine_rank(xyzs, ndecimals=ndecimals)
    return affine_rank <=1

def is_coplanar(vertex_list: list[topobj.Vertex], atol: float = None, rtol: float = None) -> bool:
    """
    This function checks if the list of points are coplanar. 
 
    Parameters
    ----------
    vertex_list : list[topobj.Vertex]
        array of vertices. It will also work for a 2darray of vertex list
        
    Returns
    -------
    True or False : bool
        If True the list of points are coplanar.
    """
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    is_2d = isinstance(vertex_list[0], np.ndarray | list)
    
    if not is_2d:
        xyzs = [v.point.xyz for v in vertex_list]
        
    else:
        xyzs = []
        for verts in vertex_list:
            xyz = [v.point.xyz for v in verts]
            xyzs.append(xyz)
            
    return is_coplanar_xyzs(xyzs, atol=atol, rtol=rtol)

def is_coplanar_xyzs(xyzs: np.ndarray, atol: float = None, rtol: float = None) -> bool | list[bool]:
    """
    This function checks if the list of xyzs are coplanar. (https://www.geeksforgeeks.org/program-to-check-whether-4-points-in-a-3-d-plane-are-coplanar/)
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray(shape(number of points, 3)) or np.ndarray(shape(number of sets of points, number of points, 3)). 
        This function takes multiple sets of points and check their coplanarity.
        
    Returns
    -------
    True or False : bool | list[bool]
        If True the list of points are coplanar.
    """
    def a_coplanar(xyzs):
        shape = np.shape(xyzs)
        is_this_collinear = True
        abcd = None
        max_cnt = shape[0] - 3
        cnt = 0
        while is_this_collinear:
            if cnt != 0:
                xyzs_roll = np.roll(xyzs, -1, axis=0)
            else:
                xyzs_roll = xyzs
            xyzs_reshape = np.reshape(xyzs_roll, (1, shape[0], shape[1]))
            abcd = planes_frm_3pts(xyzs_reshape, atol=atol, rtol=rtol)[0]
            is_this_collinear = np.isnan(abcd[0])
            
            if cnt == max_cnt and is_this_collinear == True:
                abcd = None
                is_this_collinear = False
            cnt +=1
        
        if abcd is None:
            return None
        else:
            a = abcd[0]
            b = abcd[1]
            c = abcd[2]
            d = abcd[3]
            e = a * xyzs[3:, 0] + b * xyzs[3:, 1] + c * xyzs[3:, 2] + d
            # print(e)
            is_close = np.isclose(e, 0, atol=atol, rtol=rtol)
            if False in is_close:
                return False
            else:
                return True
        
    # old codes
    # affine_rank = _affine_rank(xyzs)
    # return affine_rank <=2
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    flat = list(chain.from_iterable(xyzs))
    flat_shp = np.shape(flat)
    nshape = len(flat_shp)
    if nshape == 1:  # only 1 set of points
        if type(xyzs) != np.ndarray:
            xyzs = np.array(xyzs)
        
        return a_coplanar(xyzs)

    elif nshape == 2: #multiple sets of points
        # is_hmgnz = utility.check_2dlist_is_hmgnz(xyzs)
        # if is_hmgnz == False:
        #     are_xyzs_coplanar = [a_coplanar(np.array(xyz)) for xyz in xyzs]
        #     return np.array(are_xyzs_coplanar)
        # else:
        #     if type(xyzs) != np.ndarray:
        #         xyzs = np.array(xyzs)
        #     shape = np.shape(xyzs)
        #     print(xyzs)
        #     abcds = planes_frm_3pts(xyzs)
        #     abcdsT = abcds.T
        #     a = np.reshape(abcdsT[0], (shape[0],1))
        #     b = np.reshape(abcdsT[1], (shape[0],1))
        #     c = np.reshape(abcdsT[2], (shape[0],1))
        #     d = np.reshape(abcdsT[3], (shape[0],1))

        #     e = a * xyzs[:, 3:, 0] + b * xyzs[:, 3:, 1] + c * xyzs[:, 3:, 2] + d
        #     is_close = np.isclose(e, 0, atol=atol, rtol=rtol)
        #     res = []
        #     for ic in is_close:
        #         if False in ic:
        #             res.append(False)
        #         else:
        #             res.append(True)
        #     return np.array(res)
        are_xyzs_coplanar = [a_coplanar(np.array(xyz)) for xyz in xyzs]
        return np.array(are_xyzs_coplanar)

def is_xyz_in_bbox(xyz: np.ndarray, bbox: utility.Bbox, zdim: bool = True) -> bool:
    """
    This function check if a point is in bounding box.  
 
    Parameters
    ----------
    xyz : np.ndarray
        array defining the point np.ndarray(shape(3)).
        
    bbox : utility.Bbox
        bbox object
        
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
    bbox_arr = bbox.bbox_arr
    mnx = bbox_arr[0]
    mny = bbox_arr[1]
    mnz = bbox_arr[2]
    mxx = bbox_arr[3]
    mxy = bbox_arr[4]
    mxz = bbox_arr[5]
    
    in_bdry = False
    if zdim == True:
        if mnx<=x<=mxx and mny<=y<=mxy and mnz<=z<=mxz:
            in_bdry = True
        
    else:
        if mnx<=x<=mxx and mny<=y<=mxy:
            in_bdry = True
    
    return in_bdry

def is_anticlockwise(xyzs: np.ndarray, ref_vec: np.ndarray) -> bool:
    """
    This function checks if the list of points are arranged anticlockwise in regards to the ref_pyvec by calculating the winding number. When the number is negative they are clockwise.
    The ref_pyvec must be perpendicular to the points. 
 
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)] or np.ndarray[shape(number of sets of points, number of points in each set, 3)], array defining the points.
    
    ref_vec : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of sets of points, 3)], reference vector must be perpendicular to the list of points.
        
    Returns
    -------
    True or False : bool
        If True the list of points are arranged in anticlockwise manner, if False they are not.
    """
    shape = np.shape(xyzs)
    nshape = len(shape)
    if nshape == 1:
        raise ValueError("xyzs is a 1d array. This sometimes happens if it is a 2d arrays have arrays of different length, this is not allowed")
        
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    if type(ref_vec) != np.ndarray:
        ref_vec = np.array(ref_vec)
    
    if nshape == 2: #only 1 set of points
        xyzs = np.reshape(xyzs, (1, shape[0], shape[1]))
        res = winding_number(xyzs, ref_vec)[0]
        if res < 0:
            return False
        elif res > 0:
            return True
        else: #if res == 0 means the points are collinear
            return None
        
    elif nshape == 3:
        res = winding_number(xyzs, ref_vec)
        cond1 = np.logical_not(res <= 0) # bigger than 0 means its anticlockwise
        cond2 = np.logical_not(res != 0) #find where are the 0
        cond3 = np.where(cond2, None, cond1) #use the 0 cond rule to change the 0 to None    
        return cond3

def lineedge_intersect(edge_list1: list[topobj.Edge], edge_list2: list[topobj.Edge], atol: float = None, rtol: float = None) -> list[topobj.Vertex]:
    """
    Find the intersections between the edge_list1 and edge_list2. The edges need to only have simple lines as curve geometry
    
    Parameters
    ----------
    edge_list1 : list[topobj.Edge]
        array of edges [edge1, edge2, ...].
    
    edge_list2 : list[topobj.Edge]
        array of edges [edge1, edge2, ...].
    
    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.

    Returns
    -------
    intersections : list[topobj.Vertex]
        array of all the intersection points.
    """
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    linexyzs1 = _extract_xyzs_from_lineedge(edge_list1)
    linexyzs2 = _extract_xyzs_from_lineedge(edge_list2)
    #TODO implement for more complex polygon curve can be achieve by breaking each curve into lines
    int_pts = linexyzs_intersect(linexyzs1, linexyzs2, atol=atol, rtol=rtol)
    int_ptsT = int_pts.T
    cond1 = np.isnan(int_ptsT[0])

    vlist = []
    for cnt,cond in enumerate(cond1):
        if cond:
            vlist.append(None)
        else:
            v = create.vertex(int_pts[cnt])
            vlist.append(v)
 
    return vlist

def linexyzs_from_t(ts: list[float], linexyzs: np.ndarray) -> np.ndarray:
    """
    get a pointxyz on a line with the parameter t. Based on this post https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
    
    Parameters
    ----------
    ts : list[float]
        the t parameter, 0-1.
    
    linexyzs : np.ndarray
        array of lines [line1, line2, ...]. Each line is define as [[x1,y1,z1],[x2,y2,z2]]. np.ndarray(shape(number of lines, 2, 3))
        
    Returns
    -------
    xyzs : np.ndarray
        array of the xyz points on the line with parameter t. np.ndarray(shape(number of ts, 3))
    """
    if type(ts) != np.ndarray:
        ts = np.array(ts)
        
    if type(linexyzs) != np.ndarray:
        linexyzs = np.array(linexyzs)
    
    shape = np.shape(linexyzs)
    xyzs= np.reshape(linexyzs, (shape[0]*shape[1], shape[2]))
    xyzsT = xyzs.T
    xs1 = xyzsT[0][0::2]
    xs2 = xyzsT[0][1::2]
    
    ys1 = xyzsT[1][0::2]
    ys2 = xyzsT[1][1::2]
    
    zs1 = xyzsT[2][0::2]
    zs2 = xyzsT[2][1::2]
    
    xs_t = xs1 + ((xs2-xs1) * ts)
    ys_t = ys1 + ((ys2-ys1) * ts)
    zs_t = zs1 + ((zs2-zs1) * ts)
    
    xyzsT2 = np.array([xs_t, ys_t, zs_t])
    xyzs_t = xyzsT2.T
    
    return xyzs_t

def linexyzs_intersect(linexyzs1: np.ndarray, linexyzs2: np.ndarray, atol: float = None, rtol: float = None) -> np.ndarray:
    """
    Find the intersections between the lines. Intersection does not include when the end point of one of the line is touching the other line.
    
    Parameters
    ----------
    linexyzs1 : np.ndarray
        np.ndarray(shape(number of lines, 2, 3))

    linexyzs2 : np.ndarray
        np.ndarray(shape(number of lines, 2, 3))
    
    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.

    Returns
    -------
    intersections : np.ndarray
        np.ndarray(shape(number of intersection points, 3)), array of all the intersection points.
    """
    if type(linexyzs1) != np.ndarray:
        linexyzs1 = np.array(linexyzs1)
    
    if type(linexyzs2) != np.ndarray:
        linexyzs2 = np.array(linexyzs2)

    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    shape1 = np.shape(linexyzs1)
    shape2 = np.shape(linexyzs2)
    
    if shape1 != shape2:
        raise ValueError("Both linexyzs have to be in the same shape")
    
    xyzs1 = np.reshape(linexyzs1, (shape1[0]*shape1[1], shape1[2]))
    xyzs1_1 = xyzs1[0::2]
    xyzs1_2 = xyzs1[1::2]
    
    xyzs2 = np.reshape(linexyzs2, (shape2[0]*shape2[1], shape2[2]))
    xyzs2_1 = xyzs2[0::2]
    xyzs2_2 = xyzs2[1::2]
    
    vector_ab = xyzs2_1 - xyzs1_1
    linexyzs1_dir = xyzs1_2  - xyzs1_1 
    linexyzs2_dir = xyzs2_2  - xyzs2_1 
    vector_perpendicular = cross_product(linexyzs1_dir, linexyzs2_dir)
    
    num = cross_product(vector_ab, linexyzs2_dir)
    num = dot_product(num, vector_perpendicular)
    naxis = len(np.shape(vector_perpendicular))
    denom = np.linalg.norm(vector_perpendicular, ord = 2, axis = naxis-1, keepdims=False)
    denom = denom**2
    # cond1 = np.logical_not(num==0)
    cond1 = np.logical_not(np.isclose(num, 0))
    num = np.where(cond1, num, np.nan)
    denom = np.where(cond1, denom, np.nan)
    ts = np.where(cond1, num/denom, np.nan)
    shape3 = np.shape(linexyzs1_dir)
    ts = np.reshape(ts, (shape3[0],1))
    cond2 = np.logical_and(ts >= 0, ts <= 1)
    vector_a_scaled = linexyzs1_dir * ts
    int_pts = xyzs1_1 + vector_a_scaled
    int_pts = np.where(cond2, int_pts, np.nan)
    coplanar_xyzs = np.append(linexyzs1, linexyzs2, axis=1)
    is_planar = is_coplanar_xyzs(coplanar_xyzs)
    is_planar = np.reshape(is_planar, (shape3[0],1))
    int_pts = np.where(is_planar, int_pts, np.nan)
    dist = dist_pointxyzs2linexyzs(int_pts, linexyzs2, int_pts = False)
    # print('dists', dist)
    is_on_line2 = np.isclose(dist, 0, atol = atol, rtol=rtol)
    # print(is_on_line2)
    is_on_line2 = np.reshape(is_on_line2, (shape3[0],1))
    int_pts = np.where(is_on_line2, int_pts, np.nan)

    return int_pts

def match_xyzs_2_bboxes(xyzs: np.ndarray, bbox_list: list[utility.Bbox], zdim: bool = True) -> np.ndarray:
    """
    This function returns the point indices follow by the bbox indices which it is contained in.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)] array defining the points.
    
    bbox_list : list[utility.Bbox]
        A list of bbox
        
    zdim : bool, optional
        If True will check the z-dimension.

    Returns
    -------
    point_bbox_indices : np.ndarray
        np.ndarray(shape(2, number_of_matches)), point indices follow by the bbox indices. The two arrays corresponds to each other.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
        
    def reshape_bdry(nparray, repeat):
        arrx = np.expand_dims(nparray, axis=0)
        arrx = np.repeat(arrx, repeat, axis=0)
        return arrx
    
    npts = len(xyzs)
    
    bbox_arr_ls = np.array([bbox.bbox_arr for bbox in bbox_list])
    tbdry = bbox_arr_ls.T
    
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
        cond1 = np.logical_and(x_valid, y_valid)
        xyz_valid = np.logical_and(cond1, z_valid)
    else:
        xyz_valid = np.logical_and(x_valid, y_valid)
    
    index = np.where(xyz_valid)
    index = np.array(index)
    return index

def move_xyzs(xyzs: np.ndarray | list, directions: np.ndarray | list, magnitudes: np.ndarray | list) -> np.ndarray:
    """
    Calculates the moved xyzs based on a direction vectors and magnitudes
 
    Parameters
    ----------    
    xyzs : np.ndarray
        array defining the points np.ndarray(shape(number of points, 3)).
    
    directions : np.ndarray
        array defining the directions np.ndarray(shape(number of points, 3)).
    
    magnitudes : np.ndarray
        array defining the magnitude to move np.ndarray(shape(number of points)).
        
    Returns
    -------
    moved_xyzs : np.ndarray
        the moved xyzs np.ndarray(shape(number of points, 3)).
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

def normalise_vectors(vector_list: np.ndarray) -> np.ndarray:
    """
    This function normalise the vectors.
 
    Parameters
    ----------    
    vector_list : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)], numpy array of the vectors.
 
    Returns
    -------
    normalise_vector : np.ndarray
        normalise vectors np.ndarray[shape(3)] or np.ndarray[shape(number of vectors, 3)].
    """
    if type(vector_list) != np.ndarray:
        vector_list = np.array(vector_list)
    naxis = len(vector_list.shape)
    normalized = vector_list/np.linalg.norm(vector_list, ord = 2, axis = naxis-1, keepdims=True)
    return normalized

def planes_frm_pointxyzs_nrmls(xyzs: np.ndarray, nrmlxyzs = np.ndarray) -> np.ndarray:
    """
    calculate the d coefficient of a plane.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)]. Points on the planes.
    
    nrmlxyzs : np.ndarray
        np.ndarray[shape(number of normals, 3)]. Normals of the planes.
        
    Returns
    -------
    np.ndarray
        np.ndarray[shape(number of planes, 4)], a,b,c,d of the planes.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    if type(nrmlxyzs) != np.ndarray:
        nrmlxyzs = np.array(nrmlxyzs)

    d = (-1 * nrmlxyzs[:, 0] * xyzs[:, 0]) - (nrmlxyzs[:, 1] * xyzs[:, 1]) - (nrmlxyzs[:, 2] * xyzs[:, 2])
    d = np.reshape(d, [len(d),1])
    abcds = np.append(nrmlxyzs, d, axis=1)
    return abcds

def planes_frm_3pts(xyzs: np.ndarray, atol: float = None, rtol: float = None) -> np.ndarray:
    """
    Returns coefficient a,b,c,d for the plane equation ax + by + cz + d = 0.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray(shape(number of sets of points, 3, 3)).
    
    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance.

    Returns
    -------
    np.ndarray
        np.ndarray(shape(number of sets of points, 4)). a,b,c,d of the planes.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    a1 = xyzs[:, 1, 0] - xyzs[:, 0, 0]
    b1 = xyzs[:, 1, 1] - xyzs[:, 0, 1]
    c1 = xyzs[:, 1, 2] - xyzs[:, 0, 2]
    a2 = xyzs[:, 2, 0] - xyzs[:, 0, 0]
    b2 = xyzs[:, 2, 1] - xyzs[:, 0, 1]
    c2 = xyzs[:, 2, 2] - xyzs[:, 0, 2]
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    # check for cases where the given points are collinear
    ais0 = np.isclose(a, 0, rtol=rtol, atol=atol)
    bis0 = np.isclose(b, 0, rtol=rtol, atol=atol)
    cis0 = np.isclose(c, 0, rtol=rtol, atol=atol)
    is01 = np.logical_and(ais0, bis0)
    is0 = np.logical_and(is01, cis0)
    is0x3 = np.repeat(is0, 3)
    is0x3 = np.reshape(is0x3, (len(is0), 3))
    # normalise the vectors
    nrmls = np.vstack((a, b, c))
    nrmls = nrmls.T
    nrmls = np.where(is0x3, np.nan, nrmls)
    nrmls = normalise_vectors(nrmls)
    abc = nrmls.T
    a = abc[0]
    b = abc[1]
    c = abc[2]

    ax = np.where(is0, np.nan, -1 * a * xyzs[:, 0, 0])
    bx = np.where(is0, np.nan, b * xyzs[:, 0, 1])
    cx = np.where(is0, np.nan, c * xyzs[:, 0, 2])
    d = ax - bx -cx
    
    abcd = np.vstack((abc, d))
    abcdT = abcd.T
    return abcdT

def polygons_clipping(clipping_poly: topobj.Face, subject_poly: topobj.Face, boolean_op: str, atol: float = None, rtol: float = None) -> list[topobj.Face]:
    """
    Find the intersections between the clipping polys and the subject polys. Both the polygons cannot have holes. The result of the operation cannot have holes.
    
    Parameters
    ----------
    clipping_poly : topobj.Face
        Clipping face.
    
    subject_poly : topobj.Face
        Subject face.
    
    boolean_op : str
        can be either 'union', 'intersection', 'clip_not_subject', 'subject_not_clip'.    

    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance. 

    Returns
    -------
    intersect : list[topobj.Face]
        results of the boolean operation. None return when operations do not produce any faces.
    """
    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    clip_hole_verts = get.hole_vertices_frm_face(clipping_poly)
    subj_hole_verts = get.hole_vertices_frm_face(subject_poly)
    if len(clip_hole_verts) != 0 or len(subj_hole_verts) != 0:
        print('polygon faces have holes!! Holes will be ignored')
    
    clip_normal = get.face_normal(clipping_poly)
    clip_verts = get.bdry_vertices_frm_face(clipping_poly)
    subj_verts = get.bdry_vertices_frm_face(subject_poly)
    clip_xyzs = np.array([vert.point.xyz for vert in clip_verts]).astype(float)
    subj_xyzs = np.array([vert.point.xyz for vert in subj_verts]).astype(float)
    
    clip_res_xyzs = polyxyzs_clipping(clip_xyzs, subj_xyzs, clip_normal, boolean_op, atol=atol, rtol=rtol)
    if clip_res_xyzs != None:
        intfs = []
        for clip in clip_res_xyzs:
            cv = create.vertex_list(clip)
            int_f = create.polygon_face_frm_verts(cv)
            intfs.append(int_f)
        return intfs
    else:
        return None

def polyxyzs_clipping(clipping_polyxyzs: np.ndarray, subject_polyxyzs: np.ndarray, ref_vec: list[float] | np.ndarray, boolean_op: str,
                      atol: float = None, rtol: float = None) -> np.ndarray:
    """
    - Perform Greiner-Hormann polygon clipping. Both the polygons cannot have holes. The result of the operation cannot have holes.
    - (https://en.wikipedia.org/wiki/Greiner%E2%80%93Hormann_clipping_algorithm) (https://davis.wpi.edu/~matt/courses/clipping/)
    
    Parameters
    ----------
    clipping_polyxyz : np.ndarray
        np.ndarray(shape(number of points in polygon, 3)). The clipping polygon.
    
    subject_polyxyz : np.ndarray
        np.ndarray(shape(number of points in polygon, 3)). The clipping polygons.

    ref_vec : np.ndarray | list[float]
        list(shape(3)). The normal of the clipping polygon.
    
    boolean_op : str
        can be either 'union', 'intersection', 'clip_not_subject', 'subject_not_clip'.

    atol: float, optional
        absolute tolerance. 

    rtol: float, optional
        relative tolerance. 
    
    Returns
    -------
    intersections : np.ndarray
        np.ndarray(shape(number of polygons, number of points in polygon, 3)), array of all the clipped polygons.
    """
    def calc_alpha(edge_idxs, intxs, polyxyzs):
        dup_idxs = utility.id_dup_indices_1dlist(edge_idxs)
        # print(dup_idxs)
        if len(dup_idxs) !=0:
            new_intxs = []
            new_edge_idxs = []
            dup_flat = []
            for dup in dup_idxs:
                orig_idx = edge_idxs[dup[0]]
                dup_edge_idxs = np.take(edge_idxs, dup, axis=0)
                start_xyz = np.array([polyxyzs[orig_idx - 1]])
                end_xyz = polyxyzs[orig_idx]
                ttl_dist = dist_btw_xyzs(start_xyz[0], end_xyz)
                start_xyzs = np.repeat(start_xyz, len(dup), axis=0)
                intxs_alpha = np.take(intxs, dup, axis = 0)
                dists = dist_btw_xyzs(start_xyzs, intxs_alpha)
                alphas = dists/ttl_dist
                alphas_sort_id = np.argsort(alphas)
                intxs_alpha_sorted = intxs_alpha[alphas_sort_id].tolist()
                new_intxs.extend(intxs_alpha_sorted)
                new_edge_idxs.extend(dup_edge_idxs)
                dup_flat.extend(dup)

            all_idxs = list(range(len(edge_idxs)))
            non_dup = utility.find_xs_not_in_ys(dup_flat, all_idxs)
            if len(non_dup) > 0:
                intxs_ndup = np.take(intxs, non_dup, axis = 0)
                ndup_edge_idxs = np.take(edge_idxs, non_dup, axis=0)
                new_intxs.extend(intxs_ndup)
                new_edge_idxs.extend(ndup_edge_idxs)

            return np.array(new_intxs), np.array(new_edge_idxs)
        else:
            return intxs, edge_idxs
    
    def gen_clip(subj_intxs, subj_intxs_idxs, clip_intxs, clip_intxs_idxs, n_intxs, entry_exit, entry_exit_idxs, 
                 map_clip2subj_idxs_entry_exit, boolean_op):
        clip_polys = []
        not_complete = True
        curr_ls = None
        mv_dir = None
        nclip_pts = len(clip_intxs)
        nsubj_pts = len(subj_intxs)
        ttl_pts = (nclip_pts + nsubj_pts) * 2
        # True = Exit, False = Entry
        if entry_exit[0]:
            curr_id = subj_intxs_idxs[0]
            xyz = subj_intxs[curr_id]
            if boolean_op == 'intersection':
                mv_dir = 'backward'
            elif boolean_op == 'union':
                mv_dir = 'forward'
            elif boolean_op == 'clip_not_subject':
                mv_dir = 'backward'
            elif boolean_op == 'subject_not_clip':
                mv_dir = 'forward'
            curr_ls = 'subj'
        else:
            curr_id = clip_intxs_idxs[0]
            xyz = clip_intxs[curr_id]
            if boolean_op == 'intersection':
                mv_dir = 'forward'
            elif boolean_op == 'union':
                mv_dir = 'backward'
            elif boolean_op == 'clip_not_subject':
                mv_dir = 'backward'
            elif boolean_op == 'subject_not_clip':
                mv_dir = 'forward'
            curr_ls = 'clip'

        visited_nodes = [0]

        clip_polys.append([])
        clip_polys[-1].append(xyz.tolist())
        cnt = 0
        while not_complete:
            if mv_dir == 'backward':
                nxt_id = curr_id - 1

            elif mv_dir == 'forward':
                nxt_id = curr_id + 1

            if curr_ls == 'clip':
                if nxt_id < 0:
                    nxt_id = len(clip_intxs) - 1
                elif nxt_id > len(clip_intxs) - 1:
                    nxt_id = 0

                # print('in clip next id', nxt_id)
                xyz = clip_intxs[nxt_id]
                if nxt_id in clip_intxs_idxs: # that means will need to change ls
                    idx = np.where(clip_intxs_idxs == nxt_id)[0][0]
                    subj_idx = map_clip2subj_idxs_entry_exit[idx]

                    if subj_idx not in visited_nodes:
                        visited_nodes.append(subj_idx)
                    else: # means this is a repeated node
                        # u need to move on if there is still unvisited node
                        unvisited_nodes = utility.find_xs_not_in_ys(entry_exit_idxs, visited_nodes)
                        if len(unvisited_nodes) > 0:
                            subj_idx = unvisited_nodes[0]

                    if entry_exit[subj_idx]:
                        if boolean_op == 'intersection':
                            mv_dir = 'backward'
                        elif boolean_op == 'union':
                                mv_dir = 'forward'
                        elif boolean_op == 'clip_not_subject':
                            mv_dir = 'backward'
                        elif boolean_op == 'subject_not_clip':
                            mv_dir = 'forward'

                        curr_ls = 'subj'
                        nxt_id = map_clip2subj_idxs[idx]

            elif curr_ls == 'subj':
                if nxt_id < 0:
                    nxt_id = len(subj_intxs) - 1
                elif nxt_id > len(subj_intxs) - 1:
                    nxt_id = 0

                xyz = subj_intxs[nxt_id]
                if nxt_id in subj_intxs_idxs: # that means will need to change ls
                    idx = np.where(subj_intxs_idxs == nxt_id)[0][0]
                    if idx not in visited_nodes:
                        visited_nodes.append(idx)
                    else: # means this is a repeated node
                        # u need to move on if there is still unvisited node
                        unvisited_nodes = utility.find_xs_not_in_ys(entry_exit_idxs, visited_nodes)
                        if len(unvisited_nodes) > 0:
                            idx = unvisited_nodes[0]

                    if entry_exit[idx] == False:
                        if boolean_op == 'intersection':
                            mv_dir = 'forward'
                        elif boolean_op == 'union':
                            mv_dir = 'backward'
                        elif boolean_op == 'clip_not_subject':
                            mv_dir = 'backward'
                        elif boolean_op == 'subject_not_clip':
                            mv_dir = 'forward'
                        
                        curr_ls = 'clip'
                        nxt_id = map_subj2clip_idxs[idx]

            if len(clip_polys[-1]) == 0:
                poly_closed = False
                clip_polys[-1].append(xyz.tolist())
            else:
                if np.array_equal(xyz, clip_polys[-1][0]):
                    # the polygon is closed
                    poly_closed = True
                    if len(visited_nodes) != n_intxs:
                        clip_polys.append([])
                else:
                    poly_closed = False
                    clip_polys[-1].append(xyz.tolist())

            curr_id = nxt_id
            if len(visited_nodes) == n_intxs and poly_closed:
                not_complete = False

            elif len(visited_nodes) == n_intxs and poly_closed == False and cnt == ttl_pts:
                not_complete = False
                print('Error the result of the boolean probably have holes')
                clip_polys = None
            
            elif cnt == ttl_pts:
                not_complete = False
                print('Error unable to find boolean polygons')
                clip_polys = None

            cnt+=1
        return clip_polys
    
    if type(clipping_polyxyzs) != np.ndarray:
        clipping_polyxyzs = np.array(clipping_polyxyzs).astype(float)
        
    if type(subject_polyxyzs) != np.ndarray:
        subject_polyxyzs = np.array(subject_polyxyzs).astype(float)

    if atol is None:
        atol = settings.ATOL
    
    if rtol is None:
        rtol = settings.RTOL

    is_ccw = is_anticlockwise(subject_polyxyzs, ref_vec)
    if is_ccw:
        subject_polyxyzs = np.flip(subject_polyxyzs, axis=0)

    # check if both polygons are coplanar, if not just return no intersection
    all_xyzs = np.append(clipping_polyxyzs, subject_polyxyzs, axis = 0)
    is_pts_coplanar = is_coplanar_xyzs(all_xyzs, atol=atol, rtol=rtol)
    if is_pts_coplanar == False:
        return None
    # turn each polygon into edges
    clip_edges = _polyxyzs2edges(clipping_polyxyzs)
    subj_edges = _polyxyzs2edges(subject_polyxyzs)
    n_clip_edges = len(clip_edges)
    n_subj_edges = len(subj_edges)
    # intersect the edges and find intersections
    clip_edges_rep = np.repeat(clip_edges, n_subj_edges, axis=0)
    subj_edges_rep = np.tile(subj_edges, (n_clip_edges, 1, 1))
    intxs = linexyzs_intersect(clip_edges_rep, subj_edges_rep, atol=atol, rtol=rtol)
    if len(intxs) == 0: # if no intersection no clipping
        return None 
    # arrange each intersections into the polygon xyzs
    # forgot to calculate for alpha value
    intxsT = intxs.T
    intxsT0 = intxsT[0]
    cond1 = np.isnan(intxsT0)
    cond1 = np.logical_not(cond1)
    intx_idxs = np.where(cond1)[0]
    intxs_take = np.take(intxs, intx_idxs, axis=0)
    #-----------------------------------------------------------------------------------
    clip_edge_idxs = np.floor(intx_idxs/n_subj_edges).astype(int) + 1 # position in clip
    subj_edge_idxs = intx_idxs%n_subj_edges + 1 # positions in subj
    clip_intxs_take, clip_edge_idxs = calc_alpha(clip_edge_idxs, intxs_take, clipping_polyxyzs)
    subj_intxs_take, subj_edge_idxs = calc_alpha(subj_edge_idxs, intxs_take, subject_polyxyzs)

    clip_intxs = np.insert(clipping_polyxyzs, clip_edge_idxs, clip_intxs_take, axis = 0)
    subj_intxs = np.insert(subject_polyxyzs, subj_edge_idxs, subj_intxs_take, axis = 0)

    clip_intxs_idxs = find_these_xyzs_in_xyzs([clip_intxs_take], [clip_intxs], atol=atol, rtol=rtol)[1]
    subj_intxs_idxs = find_these_xyzs_in_xyzs([subj_intxs_take], [subj_intxs], atol=atol, rtol=rtol)[1]

    map_subj2clip_idxs = find_these_xyzs_in_xyzs([subj_intxs_take], [clip_intxs], atol=atol, rtol=rtol)[1]
    map_clip2subj_idxs = find_these_xyzs_in_xyzs([clip_intxs_take], [subj_intxs], atol=atol, rtol=rtol)[1]

    # find the entry or exit status of each intersection
    subj_intxs_idxs_bef = subj_intxs_idxs - 1
    subj_intxs_bef = np.take(subj_intxs, subj_intxs_idxs_bef, axis = 0)
    entry_exit = are_xyzs_in_polyxyzs([subj_intxs_bef], [clipping_polyxyzs], atol=atol, rtol=rtol)[0] # True = Exit, False = Entry
    entry_exit_idxs = list(range(len(entry_exit)))
    map_clip2subj_idxs_entry_exit = find_these_xyzs_in_xyzs([clip_intxs_take], [subj_intxs_take], atol=atol, rtol=rtol)[1]

    # find the boolean result polygons
    n_intxs = len(intxs_take)
    if boolean_op not in ['intersection', 'union', 'clip_not_subject', 'subject_not_clip']:
        print("Error!! Boolean operation not recognize. Please specify either 'intersection', 'union', 'clip_not_subject' or 'subject_not_clip'")
        return None
    
    clip_polys = gen_clip(subj_intxs, subj_intxs_idxs, clip_intxs, clip_intxs_idxs, n_intxs, entry_exit, 
                          entry_exit_idxs, map_clip2subj_idxs_entry_exit, boolean_op)
    
    return clip_polys

def proj_ptxyzs_on_plane(pointxyzs: np.ndarray, planexyzs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    project pts onto the plane.
    
    Parameters
    ----------
    pointxyzs : np.ndarray
        np.ndarray[shape(number of points, 3)],
    
    planexyzs : np.ndarray
        np.ndarray[shape(number of planes, 4)]. 
        
    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - dist: np.ndarray[shape(number of points)], distances.
        - points on planes : np.ndarray[shape(number of points, 3)].
    """
    nrmlxyzs = planexyzs[:, 0:3]
    dist = dot_product(nrmlxyzs, pointxyzs) + planexyzs[:, 3]
    rev_nrmls = reverse_vectorxyz(nrmlxyzs)
    ptxyz_on_pl = move_xyzs(pointxyzs, rev_nrmls, dist)
    abs_dist = np.abs(dist)
    return abs_dist, ptxyz_on_pl

def rays_bboxes_intersect(ray_list: list[utility.Ray], bbox_list: list[utility.Bbox], 
                          ndecimals: int = None) -> tuple[list[utility.Ray], list[utility.Ray], list[utility.Bbox], list[utility.Bbox]]:
    """
    This function intersect multiple rays with multiple bboxes
 
    Parameters
    ----------
    ray_list : list[utility.Ray]
        array of ray objects
        
    bbox_list : list[utility.Bbox]
        A list of bbox
        
    ndecimals: int, optional
        precision for the calculation. How many decimal place to round the calculations.

    Returns
    -------
    hit_rays : list[utility.Ray]
        rays that hit faces with new attributes documenting the faces and intersections. {'rays_bboxes_intersection': {'intersection':[], 'hit_bbox':[]}}
        
    miss_rays : list[utility.Ray]
        rays that did not hit .
        
    hit_bboxes : list[utility.Bbox]
        bboxes that are hit with new attributes documenting the intersection pt and the rays that hit it. {'rays_bboxes_intersection': {'intersection':[], 'ray':[]}}
        
    miss_faces : list[utility.Bbox]
        bboxes that are not hit by any rays.
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS
    #extract the origin and dir of the rays
    rays_xyz = np.array([[r.origin,r.dirx] for r in ray_list])
    res_pts, res_mag, res_ray_indx, res_bbox_indx = rays_xyz_bboxes_intersect(rays_xyz, bbox_list, ndecimals=ndecimals)
    # print(res_pts, res_mag, res_ray_indx, res_bbox_indx)
    if len(res_pts) != 0:
        #now i need to sort it into bboxes
        hit_bboxes = []
        miss_bboxes = []
        for cnt,bbox in enumerate(bbox_list):
            cnt_ls = np.array(cnt)
            hit_true = np.isin(res_bbox_indx, cnt_ls)
            hit_indx = np.where(hit_true)[0]
            nhits = len(hit_indx)
            if nhits == 0:
                #this bbox is not hit by any rays
                miss_bboxes.append(bbox)
            if nhits == 1:
                # print('1_hit')
                #this face is hit by one of the ray
                #find the ray and put it in its attributes
                bbox_ray = ray_list[res_ray_indx[hit_indx[0]]]
                intersect = res_pts[hit_indx[0]]
                hb_attribs = bbox.attributes
                if 'rays_bboxes_intersection' in hb_attribs:
                    ht_rf_att = hb_attribs['rays_bboxes_intersection']
                    ht_rf_att['intersection'].append(intersect)
                    ht_rf_att['ray'].append(bbox_ray)
                else:
                    attribs = {'rays_bboxes_intersection':
                                    { 'intersection': [intersect],
                                    'ray': [bbox_ray]
                                    }
                                }
                    bbox.update_attributes(attribs)
                hit_bboxes.append(bbox)
    
                #check if the attribs alr exist in the rays
                rattribs = bbox_ray.attributes
                if 'rays_bboxes_intersection' in rattribs:
                    rays_rf_att = rattribs['rays_bboxes_intersection']
                    rays_rf_att['intersection'].append(intersect)
                    rays_rf_att['hit_bbox'].append(bbox)
                else:
                    rattribs = {'rays_bboxes_intersection':
                                    {'intersection': [intersect],
                                      'hit_bbox': [bbox]}
                                }
                    bbox_ray.update_attributes(rattribs)
            elif nhits > 1:
                # print('multiple hits')
                #first take a look at the ray index
                #find the rays that are hitting this bbox
                pts_bbox = np.take(res_pts, hit_indx, axis=0)
                raysIds_bbox = np.take(res_ray_indx, hit_indx, axis=0)
                #check if they are instances where a single ray is hitting the same surfaces >1
                idDups = utility.id_dup_indices_1dlist(raysIds_bbox)
                all_indx = np.arange(len(raysIds_bbox))
                if len(idDups) > 0:
                    print('SOMETHING IS WRONG A RAY SHOULD NOT HIT THE BBOX TWICE')
                else:
                    nonDup_idxs = all_indx
                
                nonDup_idxs.sort()
                raysIds_bbox = np.take(raysIds_bbox, nonDup_idxs, axis=0)
                bbox_rays = np.take(ray_list, raysIds_bbox, axis=0)
                bbox_rays = bbox_rays.tolist()
                intersects = np.take(pts_bbox, nonDup_idxs, axis=0)
                intersects = intersects.tolist()
                hb_attribs = bbox.attributes
                
                #need to check is it hit by different rays before
                if 'rays_bboxes_intersection' in hb_attribs:
                    ht_rf_att = hb_attribs['rays_bboxes_intersection']
                    ht_rf_att['intersection'].extend(intersects)
                    ht_rf_att['ray'].extend(bbox_rays)
                else:
                    attribs = {'rays_bboxes_intersection':
                                    { 'intersection': intersects,
                                    'ray': bbox_rays
                                    }
                                }
                    bbox.update_attributes(attribs)
                    
                hit_bboxes.append(bbox)
                
                #check if the attribs alr exist in the rays
                for fcnt,bbox_ray in enumerate(bbox_rays):
                    rattribs = bbox_ray.attributes
                    if 'rays_bboxes_intersection' in rattribs:
                        rays_rf_att = rattribs['rays_bboxes_intersection']
                        rays_rf_att['intersection'].append(intersects[fcnt])
                        rays_rf_att['hit_bbox'].append(bbox)
                    else:
                        rattribs = {'rays_bboxes_intersection':
                                        {'intersection': [intersects[fcnt]],
                                          'hit_bbox': [bbox]}
                                    }
                        bbox_ray.update_attributes(rattribs)
                
        hit_rays = []
        miss_rays = []
        uq_hit_idx = np.unique(res_ray_indx)
        hit_rays = np.take(ray_list, uq_hit_idx , axis=0)
        rindx = np.arange(len(ray_list))
        miss_true = np.isin(rindx, uq_hit_idx)
        miss_true = np.logical_not(miss_true)
        miss_idx = np.where(miss_true)[0]
        miss_rays = np.take(ray_list, miss_idx, axis=0)
        return hit_rays, miss_rays, hit_bboxes, miss_bboxes
    
    else:
        return [], ray_list, [], bbox_list

def rays_faces_intersection(ray_list: list[utility.Ray], face_list: list[topobj.Face], 
                            ndecimals: int = None) -> tuple[list[utility.Ray], list[utility.Ray], list[topobj.Face], list[topobj.Face]]:
    """
    This function intersect multiple rays with multiple faces
 
    Parameters
    ----------
    ray_list : list[utility.Ray]
        array of ray objects
        
    face_list : list[topobj.Face]
        array of face objects
    
    ndecimals: int, optional
        precision for the calculation. How many decimal place to round the calculations.
        
    Returns
    -------
    hit_rays : list[utility.Ray]
        rays that hit faces with new attributes documenting the faces and intersections. {'rays_faces_intersection': {'intersection':[], 'hit_face':[]}}
        
    miss_rays : list[utility.Ray]
        rays that did not hit any faces.
        
    hit_face : list[topobj.Face]
        faces that are hit with new attributes documenting the intersection pt and the rays that hit it. {'rays_faces_intersection': {'intersection':[], 'ray':[]}}
        
    miss_faces : list[topobj.Face]
        faces that are not hit by any rays.
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS
    # extract the origin and dir of the rays 
    rays_xyz = np.array([[r.origin,r.dirx] for r in ray_list])
    # need to first triangulate the faces
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
    res_pts, res_mag, res_ray_indx, res_tri_indx = rays_xyz_tris_intersect(rays_xyz, tris_xyz, ndecimals=ndecimals)
    if len(res_pts) != 0:
        #now i need to sort it into faces
        hit_faces = []
        miss_faces = []
        for cnt, tidx_face in enumerate(tidx_faces):
            #if the triidx_face are in the hit tri indx, this face is hit by a ray
            hit_true = np.isin(res_tri_indx, tidx_face)
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
                face_rays = face_rays.tolist()
                intersects = np.take(pts_face, nonDup_idxs, axis=0)
                intersects = intersects.tolist()
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
        miss_true = np.isin(rindx, uq_hit_idx)
        miss_true = np.logical_not(miss_true)
        miss_idx = np.where(miss_true)[0]
        miss_rays = np.take(ray_list, miss_idx, axis=0)
        return hit_rays, miss_rays, hit_faces, miss_faces
    else:
        return [], ray_list, [], face_list

def rays_xyz_bboxes_intersect(rays_xyz: np.ndarray, bbox_list: list[utility.Bbox], 
                              ndecimals: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function intersect multiple rays with multiple bboxes. based on this https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection and this https://github.com/stackgl/ray-aabb-intersection/blob/master/index.js 
    
    Parameters
    ----------
    rays_xyz : np.ndarray
        array of rays [ray_xyz1,ray_xyz2,...]. Each ray is defined with [[origin], [direction]]. np.ndarray(shape(number of rays, 2, 3))
        
    bbox_list : list[utility.Bbox]
        A list of bbox

    ndecimals: int, optional
        precision for the calculation. How many decimal place to round the calculations.

    Returns
    -------
    intersect_pts : np.ndarray
        points of intersection. If no intersection returns an empty array
        
    magnitudes : np.ndarray
        array of magnitude of the rays to the intersection points. If no intersection return an empty array
        
    ray_index : np.ndarray
        indices of the rays. Corresponds to the intersect pts.
    
    tris_index : np.ndarray
        indices of the intersected triangles. Corresponds to the intersect pts.
    """
    if type(rays_xyz) != np.ndarray:
        rays_xyz = np.array(rays_xyz)
    
    if ndecimals is None:
        ndecimals = settings.NDECIMALS

    nrays = len(rays_xyz)
    nbbox = len(bbox_list)
    bbox_arr_ls = np.array([bbox.bbox_arr for bbox in bbox_list])
    bbox_arr_ls = np.around(bbox_arr_ls, decimals=ndecimals)
    #sort the bbox to match the number of rays, each ray to all the bboxes
    bbox_tiled = np.tile(bbox_arr_ls, (nrays,1))
    bboxT = bbox_tiled.T
    #sort the rays to match the boxs 
    rays_repeat = np.repeat(rays_xyz, nbbox, axis=0)
    rays_stacked = np.stack(rays_repeat, axis=1)
    rays_origs = np.around(rays_stacked[0], decimals=ndecimals)
    rays_dirs = np.around(rays_stacked[1], decimals=ndecimals)
    # rays_origs = rays_stacked[0]
    # rays_dirs = rays_stacked[1]
    rays_origsT = rays_origs.T
    rays_dirsT = rays_dirs.T
    rays_dirsT = np.where(rays_dirsT == 0, np.inf, rays_dirsT)
    #============================================
    #calculation to detect intersections
    rdirxs = rays_dirsT[0]
    x0_true = np.where(rdirxs == np.inf, True, False)
    txmin0 = bboxT[0] - rays_origsT[0]
    txmin = txmin0 / rdirxs
    txmax0 = bboxT[3] - rays_origsT[0]
    txmax = txmax0 / rdirxs
    
    txmin_neg = np.where(txmin0 < 0, True, False)
    txmax_neg = np.where(txmax0 < 0, True, False)
    
    txmin = np.where(x0_true, np.inf, txmin)
    txmin_ntrue = np.logical_and(x0_true, txmin_neg)
    txmin = np.where(txmin_ntrue, -np.inf, txmin)
    txmax = np.where(x0_true, np.inf, txmax)
    txmax_ntrue = np.logical_and(x0_true, txmax_neg)
    txmax = np.where(txmax_ntrue, -np.inf, txmax)
    
    txmin = np.around(txmin, decimals=ndecimals)
    txmax = np.around(txmax, decimals=ndecimals)
    
    txmin1 = txmin
    txmax1 = txmax
    txmin = np.where(txmin1 > txmax1, txmax1 , txmin1)
    txmax = np.where(txmin1 > txmax1, txmin1 , txmax1)
    
    rdirys = rays_dirsT[1]
    y0_true = np.where(rdirys == np.inf, True, False)
    tymin0 = bboxT[1] - rays_origsT[1] 
    tymin = tymin0 / rdirys
    tymax0 = bboxT[4] - rays_origsT[1]  
    tymax = tymax0 / rdirys
        
    tymin_neg = np.where(tymin0 < 0, True, False)
    tymax_neg = np.where(tymax0 < 0, True, False)
    
    tymin = np.where(y0_true, np.inf, tymin)
    tymin_ntrue = np.logical_and(y0_true, tymin_neg)
    tymin = np.where(tymin_ntrue, -np.inf, tymin)
    tymax = np.where(y0_true, np.inf, tymax)
    tymax_ntrue = np.logical_and(y0_true, tymax_neg)
    tymax = np.where(tymax_ntrue, -np.inf, tymax)
    
    tymin = np.around(tymin, decimals=ndecimals)
    tymax = np.around(tymax, decimals=ndecimals)
    tymin1 = tymin
    tymax1 = tymax
    tymin = np.where(tymin1 > tymax1, tymax1, tymin1)
    tymax = np.where(tymin1 > tymax1, tymin1, tymax1)
    #============================================
    #make the first judgement if there is any collisions
    
    condxy = np.logical_or(txmin > tymax, tymin > txmax)
    condxy = np.logical_not(condxy)
    condxy_index = np.where(condxy)[0]
    if len(condxy_index) == 0:
        #no intersections at all
        return [],[],[],[]
    
    swapx_true1 = np.logical_and(condxy, tymin > txmin)
    swapx_true2 = np.logical_and(condxy, tymax < txmax)
    txmin1 = txmin
    txmax1 = txmax
    txmin = np.where(swapx_true1, tymin , txmin1)
    txmax = np.where(swapx_true2, tymax , txmax1)
    
    #============================================
    # calcuate the z-dim
    rdirzs = rays_dirsT[2]
    z0_true = np.where(rdirzs == np.inf, True, False)
    tzmin0 = bboxT[2] - rays_origsT[2] 
    tzmin = tzmin0 / rdirzs
    tzmax0  = bboxT[5] - rays_origsT[2]
    tzmax = tzmax0 / rdirzs
    
    tzmin_neg = np.where(tzmin0 < 0, True, False)
    tzmax_neg = np.where(tzmax0 < 0, True, False)
    
    tzmin = np.where(z0_true, np.inf, tzmin)
    tzmin_ntrue = np.logical_and(z0_true, tzmin_neg)
    tzmin = np.where(tzmin_ntrue, -np.inf, tzmin)
    tzmax = np.where(z0_true, np.inf, tzmax)
    tzmax_ntrue = np.logical_and(z0_true, tzmax_neg)
    tzmax = np.where(tzmax_ntrue, -np.inf, tzmax)
    
    tzmin = np.around(tzmin, decimals=ndecimals)
    tzmax = np.around(tzmax, decimals=ndecimals)
    tzmin1 = tzmin
    tzmax1 = tzmax
    tzmin = np.where(tzmin1 > tzmax1, tzmax1 , tzmin1)
    tzmax = np.where(tzmin1 > tzmax1, tzmin1 , tzmax1)
    #============================================
    # make judgement on the zdim
    #make the first judgement if there is any collisions
    condxyz = np.logical_or(txmin > tzmax, tzmin > txmax)
    condxyz = np.logical_not(condxyz)
    condxyz = np.logical_and(condxy, condxyz)
    condxyz_index = np.where(condxyz)[0]
    if len(condxyz_index) == 0:
        #no intersections at all
        return [],[],[],[]
    
    swapz_true1 = np.logical_and(condxyz, tzmin > txmin)
    swapz_true2 = np.logical_and(condxyz, tzmax < txmax)
    txmin1 = txmin
    txmax1 = txmax
    txmin = np.where(swapz_true1, tzmin , txmin1)
    txmax = np.where(swapz_true2, tzmax , txmax1)
    
    tx_min_true = np.logical_not(txmin > 0)
    tx_min_true = np.logical_and(tx_min_true, condxyz)
    txmin = np.where(tx_min_true, txmax, txmin)
    #check again if the magnitude and the max is positive
    tx_min_true = np.logical_not(txmin < 0)
    tx_min_true = np.logical_and(tx_min_true, condxyz)
    tx_min_true_index = np.where(tx_min_true)[0]
    if len(tx_min_true_index) == 0:
        #no intersections at all
        return [],[],[],[]
    
    mags = np.where(tx_min_true, txmin, np.zeros(len(txmin)))
    # print('mags',mags)
    mags_reshape = np.reshape(mags,[len(mags),1])
    dir_mags = rays_dirs*mags_reshape
    intPts = rays_origs + dir_mags
    
    mags = np.where(tx_min_true, mags, np.full([len(mags)], np.inf))
    mags_ray_reshape = np.reshape(mags, (nrays,nbbox))
    mags_true = np.logical_not(mags_ray_reshape==np.inf)
    mags_true_index = np.where(mags_true)
    
    intPts_reshape = np.reshape(intPts, (nrays,nbbox,3))
    res_pts = intPts_reshape[mags_true_index]
    res_mag = mags_ray_reshape[mags_true_index]
    res_ray_indx = mags_true_index[0]
    res_bbox_indx = mags_true_index[1]

    res_pts, res_mag, res_ray_indx, res_bbox_indx= _acct4obstruction(res_pts, 
                                                                   res_mag, 
                                                                   res_ray_indx, 
                                                                   res_bbox_indx)
    
    return res_pts, res_mag, res_ray_indx, res_bbox_indx

def rays_xyz_tris_intersect(rays_xyz: np.ndarray, tris_xyz: np.ndarray, ndecimals: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function intersect multiple rays with multiple triangles. https://github.com/johnnovak/raytriangle-test/blob/master/python/perftest.py 
 
    Parameters
    ----------
    rays_xyz : np.ndarray
        array of rays [ray_xyz1,ray_xyz2,...]. Each ray is defined with [[origin], [direction]]. np.ndarray(shape(number of rays, 2, 3))
        
    tris_xyz : np.ndarray
        array of triangles [tri_xyz1, tri_xyz2, ...]. Each triangle is define with [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]. np.ndarray(shape(number of triangles, 3, 3))

    ndecimals: int, optional
        precision for the calculation. How many decimal place to round the calculations.

    Returns
    -------
    intersect_pts : np.ndarray
        points of intersection. If no intersection returns an empty array
        
    magnitudes : np.ndarray
        array of magnitude of the rays to the intersection points. If no intersection return an empty array
        
    ray_index : np.ndarray
        indices of the rays. Corresponds to the intersect pts.
    
    tris_index : np.ndarray
        indices of the intersected triangles. Corresponds to the intersect pts.
    """
    if ndecimals is None:
        ndecimals = settings.NDECIMALS

    dets_threshold = settings.ATOL
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
    dets = dot_product(xyzs1_0, pvecs)
    dets_round = np.around(dets, decimals=ndecimals)
    # print('dets', dets)
    dets_true = np.logical_not(dets_round<dets_threshold) #bigger than 0
    dets_index = np.where(dets_true)[0]
    # print(dets_index)
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
                  dot_product(tvecs, pvecs)*invDet, 
                  np.zeros(len(dets_true)))
    
    us = np.around(us, decimals=ndecimals)
    us_true = np.logical_and(us>=0.0-dets_threshold, us<=1.0+dets_threshold)
    us_true = np.logical_and(us_true, dets_true)
    us_index = np.where(us_true)[0]
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
                  dot_product(raysT[1], qvecs)*invDet,
                  np.zeros([len(us_true)]))
    vs = np.around(vs, decimals = ndecimals)
    vs_true1 = np.logical_not(vs < 0.0-dets_threshold)
    usvs = us+vs
    usvs = np.around(usvs, decimals=ndecimals)
    vs_true2 = np.logical_not(usvs > 1.0+dets_threshold)
    vs_true = np.logical_and(vs_true1, vs_true2)
    vs_true = np.logical_and(vs_true, us_true)
    mags = np.where(vs_true, 
                    dot_product(xyzs2_0, qvecs)*invDet,
                    np.zeros([len(vs_true)]))
    
    mags = np.around(mags, decimals=ndecimals)
    mags_true = np.logical_not(mags < dets_threshold)
    mags = np.where(mags_true, mags, np.zeros([len(vs_true)]))
    # print('mags',mags)
    mags_reshape = np.reshape(mags,[len(mags),1])
    dir_mags = raysT[1]*mags_reshape
    intPts = raysT[0] + dir_mags
    #============================================
    #separate the intersections into ray-based groups
    #sub zeros with infinity to make processing easier later
    mags = np.where(mags_true, mags, np.full([len(vs_true)], np.inf))
    mags_ray_reshape = np.reshape(mags, (nrays,ntris))
    mags_true = np.logical_not(mags_ray_reshape==np.inf)
    mags_true_index = np.where(mags_true)
    intPts_reshape = np.reshape(intPts, (nrays,ntris,3))
    res_pts = intPts_reshape[mags_true_index]
    res_mag = mags_ray_reshape[mags_true_index]
    res_ray_indx = mags_true_index[0]
    res_tri_indx = mags_true_index[1]
    res_pts, res_mag, res_ray_indx, res_tri_indx= _acct4obstruction(res_pts, 
                                                                   res_mag, 
                                                                   res_ray_indx, 
                                                                   res_tri_indx)
    
    return res_pts, res_mag, res_ray_indx, res_tri_indx

def reverse_vectorxyz(vector_xyzs: np.ndarray) -> np.ndarray:
    """
    This function calculates the reverse of the vectorxyz . 
 
    Parameters
    ----------
    vector_xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)] | np.ndarray[shape(3)].
    
    Returns
    -------
    reverse_vector : np.ndarray
        np.ndarray[shape(number of points, 3)]. The reverse of all the points.
    """
    if type(vector_xyzs) != np.ndarray:
        vector_xyzs = np.array(vector_xyzs)

    return vector_xyzs*-1

def rotate_matrice(axis: list[float], rotation: float) -> np.ndarray:
    """
    This function calculate a 4x4 translation matrice. The rotation in counter-clockwise.
 
    Parameters
    ----------    
    axis : list[float]
        the axis for rotation, must be a normalised vector.
    
    rotation : float
        the rotation in degrees.
 
    Returns
    -------
    matrice : np.ndarray
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

def scale_matrice(sx: float, sy: float, sz: float) -> np.ndarray:
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
    matrice : np.ndarray
        4x4 scaling matrice.
    """
    
    mat = np.array([[sx, 0, 0, 0],
                    [0, sy, 0, 0],
                    [0, 0, sz, 0],
                    [0, 0, 0, 1]])
    return mat

def translate_matrice(tx: float, ty: float, tz: float) -> np.ndarray:
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
    matrice : np.ndarray
        4x4 translation matrice.
    """
    mat = np.array([[1, 0, 0, tx],
                    [0, 1, 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]])
    
    return mat

def trsf_xyzs(xyzs: list, trsf_mat: np.ndarray) -> np.ndarray:
    """
    Calculates the transformed xyzs based on the given matrice
 
    Parameters
    ----------    
    xyzs : list
        list[shape(number of points, 3)] or list[shape(number of sets of points, number of points in each set, 3)], array defining the points.

    trsf_mat : np.ndarray
        np.ndarray[shape(4, 4)] or np.ndarray[shape(number of matrices, 4, 4)], 4x4 matrices to transform the corresponding xyzs.
    
    Returns
    -------
    trsf_xyzs : np.ndarray
        the transformed xyzs.
    """
    def apply_trsf(xyzs, trsf_matT):
        npos = len(xyzs)
        xyzw = np.ones((npos,4))
        xyzw[:,:-1] = xyzs
        t_xyzw = np.dot(xyzw, trsf_matT)
        return t_xyzw
        
    flat = list(chain.from_iterable(xyzs))
    flat_shp = np.shape(flat)
    nshape = len(flat_shp)
    if nshape == 1: #only 1 set of points
        if type(xyzs) != np.ndarray:
            xyzs = np.array(xyzs)
        if type(trsf_mat) != np.ndarray:
            trsf_mat = np.array(trsf_mat)
        # transform the xyzs
        trsf_xyzs = apply_trsf(xyzs, trsf_mat.T)
        trsf_xyzs = trsf_xyzs[:,:-1]
        return trsf_xyzs
    
    elif nshape == 2:
        nsets = len(xyzs)
        nmats = len(trsf_mat)
        if nsets == nmats:
            if type(trsf_mat) != np.ndarray:
                trsf_mat = np.array(trsf_mat)

            xyzs_flat = np.array(flat)
            each_set_cnt = []
            for setx in xyzs:
                each_set_cnt.append(len(setx))

            each_set_cnt = np.array(each_set_cnt)
            trsf_matT = np.transpose(trsf_mat, axes=[0, 2, 1])
            trsf_xyzs = apply_trsf(xyzs_flat, trsf_matT)
            trsf_2dlist = []
            prev_scnt = 0
            for cnt, scnt in enumerate(each_set_cnt):
                trsf_xyzs_set = trsf_xyzs[:,cnt]
                trsf_xyzs_set = trsf_xyzs_set[:,:-1]
                trsf_xyzs_set = trsf_xyzs_set[prev_scnt: prev_scnt + scnt].tolist()
                trsf_2dlist.append(trsf_xyzs_set)
                prev_scnt+=scnt
            return trsf_2dlist
        else:
            print('Error number of sets of points does not match number of transform matrices')
            return None

def winding_number(xyzs: np.ndarray, ref_vec: np.ndarray) -> int:
    """
    Calculate the winding number. 
 
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)] or np.ndarray[shape(number of sets of points, number of points in each set, 3)], array defining the points.
    
    ref_vec : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of sets of points, 3)], reference vector must be perpendicular to the list of points.
        
    Returns
    -------
    winding_number : int/list[int]
        winding number(s)
    """
    shape = np.shape(xyzs)
    xyzs2 = np.roll(xyzs, -1, axis=1)
    #reshape both of the xyzs
    xyzs_rs = np.reshape(xyzs, (shape[0]*shape[1], shape[2]))
    xyzs2_rs = np.reshape(xyzs2, (shape[0]*shape[1], shape[2]))
    prod = cross_product(xyzs_rs, xyzs2_rs)
    prod = np.reshape(prod, shape)
    ttl = np.sum(prod, axis=1)
    res = dot_product(ttl, ref_vec)
    return res

def xyzs_in_bbox(xyzs: np.ndarray, bbox: utility.Bbox, zdim: bool = True, indices: bool = False) -> np.ndarray:
    """
    are the points within the given boundary box.
 
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray[shape(number of points, 3)].
    
    bbox : utility.Bbox
        bbox object
        
    zdim : bool, optional
        If True will check the z-dimension.
        
    indices : bool, optional
        Specify whether to return the indices of the points in the boundary. If True will not return the points. Default = False.
    
    Returns
    -------
    points_in_bdry : np.ndarray
        The points that is in the boundary np.ndarray(shape(number of points, 3)). If indices==True, this will be the indices instead of the actual points.
    """
    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)
    
    zip_xyzs = xyzs.T
    xlist = zip_xyzs[0] 
    ylist = zip_xyzs[1]
    zlist = zip_xyzs[2]
    
    bbox_arr = bbox.bbox_arr
    mnx = bbox_arr[0]
    mny = bbox_arr[1]
    mxx = bbox_arr[3]
    mxy = bbox_arr[4]
    
    x_valid = np.logical_and((mnx <= xlist),
                             (mxx >= xlist))
    
    y_valid = np.logical_and((mny <= ylist),
                             (mxy >= ylist))
    
    if zdim == True:
        mnz = bbox_arr[2]
        mxz = bbox_arr[5]
        
        z_valid = np.logical_and((mnz <= zlist),
                                 (mxz >= zlist))
        cond1 = np.logical_and(x_valid, y_valid)
        cond2 = np.logical_and(cond1, z_valid)
        index_list = np.where(cond2)[0]
    
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

def xyzs_mean(xyzs: list) -> np.ndarray:
    """
    This function calculates the mean of all points. 
 
    Parameters
    ----------
    xyzs : list
        list[shape(number of points, 3)] or list[shape(number of set of points, number of points, 3)].
    
    Returns
    -------
    midpt : np.ndarray
        np.ndarray[shape(3)] or np.ndarray[shape(number of set of points, 3)] The mean of all the points.
    """
    flat = list(chain.from_iterable(xyzs))
    flat_shp = np.shape(flat)
    nshape = len(flat_shp)
    if nshape == 1: #only 1 set of points
        if type(xyzs) != np.ndarray:
            xyzs = np.array(xyzs)             
        centre_pt = np.nanmean(xyzs, axis=0)
    
    if nshape == 2:
        xyzs = modify.hmgnz_xyz2dlist(xyzs)
        centre_pt = np.nanmean(xyzs, axis=1)
    return centre_pt

def _acct4obstruction(intPts: np.ndarray, mags: np.ndarray, rayIndxs: np.ndarray, 
                     objIndxs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    accounts for obstruction when shooting rays
 
    Parameters
    ----------
    intPts : np.ndarray
        xyzs np.ndarray(shape(number of points, 3)).
    
    mags : np.ndarray
        array of magnitudes
    
    rayIndxs : np.ndarray
        array of ray indices
    
    objIndxs : np.ndarray
        array of obstruction indices
        
    Returns
    -------
    intersect_pts : np.ndarray
        points of unobstructed intersection. If no intersection returns an empty array
        
    magnitudes : np.ndarray
        array of unobstructed magnitude of the rays to the intersection points. If no intersection return an empty array
        
    ray_index : np.ndarray
        indices of the unobstructed rays. Corresponds to the intersect pts.
    
    tris_index : np.ndarray
        indices of the unobstructed obj. Corresponds to the intersect pts.
    """
    indx = np.arange(len(rayIndxs))
    dupIds = utility.id_dup_indices_1dlist(rayIndxs)
    dupIds_flat = list(chain(*dupIds))
    # print(dupIds)
    non_obs_indx = utility.find_xs_not_in_ys(indx, dupIds_flat)
    for di in dupIds:
        di.sort()
        mag = mags[di]
        index = np.where(mag == mag.min())[0]
        index = index.flatten()
        index = index + di[0]
        non_obs_indx = np.append(non_obs_indx, index)
    
    non_obs_indx.sort()
    nb_intPts = np.take(intPts, non_obs_indx, axis=0)
    nb_mags = np.take(mags, non_obs_indx, axis=0)
    nb_rayIndxs = np.take(rayIndxs, non_obs_indx, axis=0)
    nb_objIndxs = np.take(objIndxs, non_obs_indx, axis=0)
    return nb_intPts, nb_mags, nb_rayIndxs, nb_objIndxs

def _affine_rank(xyzs: np.ndarray, ndecimals = None) -> np.ndarray:
    """
    This function calculate the affine rank of the xyzs. If having a 3d array.
    
    Parameters
    ----------
    xyzs : np.ndarray
        np.ndarray(shape(number of points, 3)) or np.ndarray(shape(number of sets of points, number of points, 3)). 
        This function takes multiple sets of points and check their coplanarity.
    
    ndecimals: int, optional
        the number of decimals to round off to compare if points are the same.

    Returns
    -------
    affine_rank : np.ndarray
        affine rank of the given xyzs.
    """
    #--------------------------------------------------------------------------------------------------------------------------------
    def rank_affine(xyzs):
        centroid = xyzs_mean(xyzs)
        centered_pts = xyzs - centroid
        affine_rank = matrix_rank(centered_pts)
        return affine_rank
    #--------------------------------------------------------------------------------------------------------------------------------
    if ndecimals is None:
        ndecimals = settings.NDECIMALS

    shape = np.shape(xyzs)
    nshape = len(shape)
    if nshape == 1:
        raise ValueError(
            "this is a 1d array. This sometimes happens if arrays have arrays of different length, this is not allowed")

    if type(xyzs) != np.ndarray:
        xyzs = np.array(xyzs)

    xyzs = np.round(xyzs, decimals=ndecimals)
    if nshape == 2:  # only 1 set of points
        xyzs = np.unique(xyzs, axis=0)
        affine_rank = rank_affine(xyzs)
        return affine_rank

    elif nshape == 3:
        uniqs = [np.unique(u, axis=0) for u in xyzs]
        is_hmgnz = utility.check_2dlist_is_hmgnz(uniqs)
        if is_hmgnz == False:
            affine_rank = [rank_affine(uniq) for uniq in uniqs]
            return np.array(affine_rank)
        else:
            shape1 = np.shape(uniqs)
            uniqs = np.array(uniqs)
            centroid = xyzs_mean(uniqs)
            nrepeat = np.repeat([shape1[1]], shape1[0])
            centroid = np.repeat(centroid, nrepeat, axis=0)
            centroid = np.reshape(centroid, shape1)
            centered_pts = uniqs - centroid
            affine_rank = matrix_rank(centered_pts)
            return affine_rank

def _extract_xyzs_from_lineedge(edge_list: list[topobj.Edge]) -> np.ndarray:
    """
    extract line xyzs from edges and throw an error if the curve is not a line. A line is a curve defined by only two points
    
    Parameters
    ----------
    edge_list : list[topobj.Edge]
        array of edges [edge1, edge2, ...].
    
    Returns
    -------
    xyzs : np.ndarray
        array of edges [edge1, edge2, ...]. Each edge is define as [[x1,y1,z1],[x2,y2,z2],[x3,y3,z3]]. np.ndarray(shape(number of lines, 2, 3))
    """
    linexyzs = []
    for e in edge_list:
        vs = get.vertices_frm_edge(e)
        if len(vs) == 2:
            xyzs = [v.point.xyz for v in vs]
            linexyzs.append(xyzs)
        else:
            raise ValueError('The curve in the edge needs to be a line defined by 2 vertices')
            
    return linexyzs

def _polyxyzs2edges(polyxyzs) -> np.ndarray:
    """
    convert polyxyzs to linexyzs.

    Parameters
    ----------
    polyxyzs : np.ndarray
        np.ndarray[shape(number of polygons, number of points in polygon, 3)]. 

    Returns
    -------
    np.darray
        list[shape(number of edges in polygon, 2, 3 )].

    """
    polyxyzs1 = polyxyzs[:]
    # create edge for the polygons
    poly_edges = np.repeat(polyxyzs1, 2, axis=0)
    # roll it so the points are arrange in edges
    poly_edges = np.roll(poly_edges, -1, axis=0)
    # reshape the arrays into edges
    poly_edges = np.reshape(poly_edges, (int(len(poly_edges)/2),2,3))
    return poly_edges

def _rmv_nan_frm_hmgn_polys_edges_xyzs(poly_xyzs2d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    remove nan from polys_edges_xyzs so as to prepare for further operation e.g. line-line intersection.

    Parameters
    ----------
    poly_xyzs2d : np.ndarray
        np.ndarray[shape(number of polygons, number of points in polygon, 3)]. This array will contain np.nan in the xyzs that need to be removed due to the homogenized operation.
    
    Returns
    -------
    poly_edges_nt_nan: np.ndarray
        np.ndarray[shape(ttl number of edges in all polygons, 2, 3 )].

    is_nt_nan_id_res: np.ndarray
        np.ndarray[shape(number of edges that is not np.nan, 1 )]. The index of the edges in the polygon that are not np.nan.

    ttl_edges_res: np.ndarray
        np.ndarray[shape(number of edges, 3 )]. An empty array for putting results of future operation into the original edge shape.
    
    poly_edges_shp: np.ndarray
       shape of the original poly edges with nan.

    is_nt_nan_id: np.ndarray
       index of the original edges in the polygon that are not np.nan.
    """
    poly_edges = np.repeat(poly_xyzs2d, 2, axis=1)
    poly_edges = np.roll(poly_edges, -1, axis=1)
    # need to remove all the nan to prepare for future operations e.g. the line line intersection
    poly_edges_shp = np.shape(poly_edges)
    poly_edges_flat = np.reshape(poly_edges, [poly_edges_shp[0] * poly_edges_shp[1], poly_edges_shp[2]])
    pe_flat_T_0 = poly_edges_flat.T[0]
    is_nan = np.isnan(pe_flat_T_0)
    is_nan_id = np.where(is_nan)[0]
    is_nan_id = is_nan_id/2
    is_nan_id = np.ceil(is_nan_id) # I am not so sure if this will work everytime, this is because of the roll -1 
    is_nan_id = np.unique(is_nan_id).astype(int)
    # ---------------------------------
    is_nt_nan = np.logical_not(is_nan)
    is_nt_nan_id = np.where(is_nt_nan)[0]
    ttl_nedges = int(poly_edges_shp[0] * poly_edges_shp[1]/2)
    is_nt_nan_id_res = utility.find_xs_not_in_ys(list(range(ttl_nedges)), is_nan_id)
    is_nt_nan_id_res = np.reshape(is_nt_nan_id_res, [is_nt_nan_id_res.size, 1])

    # create a dummy result array for putting in the results later so it is easy to reshape and compute the results
    ttl_edges_res = np.empty([ttl_nedges, poly_edges_shp[2]])
    ttl_edges_res.fill(np.nan)

    poly_edges_flat_nt_nan = np.take(poly_edges_flat, is_nt_nan_id, axis=0)
    pe_nt_nan_shp = np.shape(poly_edges_flat_nt_nan)
    poly_edges_nt_nan = np.reshape(poly_edges_flat_nt_nan, [int(pe_nt_nan_shp[0]/2), 2, pe_nt_nan_shp[1]])

    return poly_edges_nt_nan, is_nt_nan_id_res, ttl_edges_res, poly_edges_shp, is_nt_nan_id

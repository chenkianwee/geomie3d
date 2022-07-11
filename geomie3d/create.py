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
from . import topobj
from . import utility

from .geomdl import BSpline, utilities, construct

def box(dimx, dimy, dimz, attributes = {}):
    """
    Constructs a box which is a solid topology where its bottom face midpt is at the origin (0,0,0).
 
    Parameters
    ----------
    dimx : float
        length of box in the x-axis.
    
    dimy : float
        length of box in the y-axis.
        
    dimz : float
        height of box.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    box : solid topology
        A box of solid topology
    """
    dimx_half = dimx/2
    dimy_half = dimy/2
    
    mnx = 0 - dimx_half
    mny = 0 - dimy_half
    mnz = 0
    
    mxx = 0 + dimx_half
    mxy = 0 + dimy_half
    mxz = dimz
    
    #bottom face
    xyz_list1 = np.array([[mnx, mny, mnz],
                          [mnx, mxy, mnz],
                          [mxx, mxy, mnz],
                          [mxx, mny, mnz]])
    
    vlist1 = vertex_list(xyz_list1)
    face1 = polygon_face_frm_verts(vlist1)
    
    #left vertical face
    xyz_list2 = np.array([[mnx, mxy, mnz],
                          [mnx, mny, mnz],
                          [mnx, mny, mxz],
                          [mnx, mxy, mxz]])
    
    vlist2 = vertex_list(xyz_list2)
    face2 = polygon_face_frm_verts(vlist2)
    
    #south vertical face
    xyz_list3 = np.array([[mnx, mny, mnz],
                          [mxx, mny, mnz],
                          [mxx, mny, mxz],
                          [mnx, mny, mxz]])
    
    vlist3 = vertex_list(xyz_list3)
    face3 = polygon_face_frm_verts(vlist3)
    
    #right vertical face
    xyz_list4 = np.array([[mxx, mny, mnz],
                          [mxx, mxy, mnz],
                          [mxx, mxy, mxz],
                          [mxx, mny, mxz]])
    
    vlist4 = vertex_list(xyz_list4)
    face4 = polygon_face_frm_verts(vlist4)
    
    #north vertical face
    xyz_list5 = np.array([[mxx, mxy, mnz],
                          [mnx, mxy, mnz],
                          [mnx, mxy, mxz],
                          [mxx, mxy, mxz]])
    
    vlist5 = vertex_list(xyz_list5)
    face5 = polygon_face_frm_verts(vlist5)
    
    #top cap face
    xyz_list6 = np.array([[mnx, mny, mxz],
                          [mxx, mny, mxz],
                          [mxx, mxy, mxz],
                          [mnx, mxy, mxz]])
    
    vlist6 = vertex_list(xyz_list6)
    face6 = polygon_face_frm_verts(vlist6)
    shell = topobj.Shell(np.array([face1, face2, face3, face4, face5, face6]))
    solid = topobj.Solid(shell)
    return solid
    
def ray(xyz_orig, xyz_dir):
    """
    This function constructs a ray object.
 
    Parameters
    ----------
    xyz_orig : tuple
        tuple with the xyz coordinates of the ray origin.
        
    xyz_dir : tuple
        tuple with the xyz of the ray direction.
 
    Returns
    -------
    ray : ray object
        A ray object
    """
    ray = utility.Ray(xyz_orig, xyz_dir)
    return ray
    
def vertex(xyz, attributes = {}):
    """
    This function constructs a vertex topology.
 
    Parameters
    ----------
    xyz : tuple
        tuple with the xyz coordinates.
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    vertex : vertex topology
        A vertex topology containing a point geometry
    """
    point = geom.Point(xyz)
    vertex = topobj.Vertex(point, attributes = attributes)
    return vertex
    
def vertex_list(xyz_list, attributes_list = []):
    """
    This function constructs a list of vertex topology.
 
    Parameters
    ----------
    xyz_list : list of tuple
        list of tuple with the xyz coordinates.
        
    attributes_list : list of dictionary, optional
        Dictionary of the attributes
 
    Returns
    -------
    vertex_list : list of vertex topology
        A list of vertex topology.
    """
    nxyz = len(xyz_list)
    natts = len(attributes_list)
    
    #check if the list match
    is_atts = False
    if natts != 0:
        is_atts = True
        if natts != nxyz:
            raise NameError("Number of xyz_list and attributes_list do not match")
    
    if is_atts:
        vlist = np.array([vertex(xyz_list[cnt], attributes = attributes_list[cnt]) 
                          for cnt in range(nxyz)])
    else:
        vlist = np.array([vertex(xyz_list[cnt]) for cnt in range(nxyz)])
    
    return vlist

def polygon_face_frm_verts(vertex_list, hole_vertex_list = [], attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology. 
        
    hole_vertex_list : a 2d list of vertices
        A 2d list of vertex topologies that is the hole of the face
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    bdry_wire = wire_frm_vertices(vertex_list)
    
    hole_wire_list = []
    for hole in hole_vertex_list:
        hole_wire =  wire_frm_vertices(hole)
        hole_wire_list.append(hole_wire)
    face = polygon_face_frm_wires(bdry_wire, hole_wire_list = hole_wire_list, attributes = attributes)
    return face

def polygon_face_frm_wires(bdry_wire, hole_wire_list = [], attributes = {}):
    """
    This function constructs a face polygon from a list of vertices.
 
    Parameters
    ----------
    bdry_wire : a wire
        A wire topology that is the boundary of the face
        
    hole_wire_list : a list of wires
        A list of wire topologies that is the hole of the face
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    polygon_face : face topology
        A face topology containing a polygonsurface geometry
    """
    face = topobj.Face(attributes = attributes)
    face.add_polygon_surface(bdry_wire, hole_wire_list = hole_wire_list)
    return face

def bspline_face_frm_ctrlpts(ctrlpts_xyz, knotvector_u, knotvector_v, deg_u, deg_v,
                             resolution = 0.06, attributes = {}):
    """
    This function creates a bspline face with control points. Refer to for more help https://github.com/orbingol/geomdl-examples/blob/master/surface/ex_surface01.py
 
    Parameters
    ----------
    ctrlpts_xyz : array of xyzs
        control points of the bspline surface. 
    
    knotvector_u : int
        knockvector_u
    
    knotvector_v : int
        knockvector_v
    
    deg_u : int
        the degree of the surface in u direction.
        
    deg_v : int
        the degree of the surface in v direction.
    
    resolution : float
        a number between 0-1. The smaller the number the higher resolution of the surface stored.
        
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Returns
    -------
    bspline_face : face topology
        A face topology containing a bspline surface geometry
    """
    # Create a BSpline surface instance
    surf = BSpline.Surface()
    # Set degrees
    surf.degree_u = deg_u
    surf.degree_v = deg_v
    kv_u = knotvector_u
    kv_v = knotvector_v
    surf.set_ctrlpts(ctrlpts_xyz, kv_u, kv_v)
    # Set knot vectors
    surf.knotvector_u = utilities.generate_knot_vector(surf.degree_u, kv_u)
    surf.knotvector_v = utilities.generate_knot_vector(surf.degree_v, kv_v)

    # Set evaluation delta
    surf.delta = resolution
    # Evaluate surface
    surf.evaluate()
    face = topobj.Face(attributes = attributes)
    face.add_bspline_surface(surf)
    return face

def bspline_face_frm_loft(bspline_edge_list,deg=1,resolution=0.06,attributes = {}):
    """
    This function creates a bspline face from bspline edges.
    
    Parameters
    ----------
    bspline_edge_list : array of Edge with bspline geometry
        array of edges with bspline geometry. 
        
    deg : int, optional
        the degree of the surface.
    
    resolution : float, optional
        a number between 0-1. The smaller the number the higher resolution of the surface stored.
        
    attributes : dictionary, optional
        dictionary of the attributes.
        
    Returns
    -------
    bspline_face : face topology
        A face topology containing a bspline surface geometry
    """
    crv_ls = [e.curve for e in bspline_edge_list]
    surf = construct.construct_surface('u', *crv_ls, degree = deg)
    # Set evaluation delta
    surf.delta = resolution
    # Evaluate surface
    surf.evaluate()
    face = topobj.Face(attributes = attributes)
    face.add_bspline_surface(surf)
    return face

def grids_frm_bspline_face(face, columns, rows, polygonise = True):
    """
    This function creates a grid from a face.
 
    Parameters
    ----------
    face : face object
        face to be gridded.
        
    columns : int
        number of columns.
        
    rows : int
        number of rows.
        
    Returns
    -------
    grid : array of faces
        the gridded face, if grid_pts=True, will return empty array.
        
    """
    flist = []
    pts = grid_pts_frm_bspline_face(face,columns+1,rows+1,xyzs=True)
    ngrids = columns*rows
    if polygonise == False:
        for cnt in range(ngrids):
            id1 = cnt + int(cnt/columns)
            id2 = id1 + 1
            id3 = id1 + columns+2
            id4 = id3 - 1
            f_xyz = [pts[id4], pts[id3], pts[id1], pts[id2]]
            f = bspline_face_frm_ctrlpts(f_xyz, 2, 2, 1, 1, resolution = 0.5)
            flist.append(f)
    else:
        for cnt in range(ngrids):
            id1 = cnt + int(cnt/columns)
            id2 = id1 + 1
            id3 = id1 + columns+2
            id4 = id3 - 1
            # create polygon surface
            f_xyz = [pts[id1], pts[id2], pts[id3], pts[id4]]
            f_v = vertex_list(f_xyz)
            f = polygon_face_frm_verts(f_v)
            flist.append(f)
    return flist
    
def grid_pts_frm_bspline_face(face,columns,rows,xyzs=False):
    """
    This function creates a grid pts from a face.
 
    Parameters
    ----------
    face : face object
        face to be gridded.
        
    columns : int
        number of columns.
        
    rows : int
        number of rows.
    
    xyzs : bool, optional
        If True return the xyzs, False return vertices.
    
    Returns
    -------    
    grid_pts : array of vertices
        grid of pts. If xyzs == True, return xyzs.
    """
    if face.surface_type == geom.SrfType.BSPLINE:
        urange = [0,1,columns]
        vrange = [0,1,rows]
        params = utility.gen_gridxyz(urange, vrange)
        surface = face.surface
        pts = surface.evaluate_list(params)
        if xyzs == True:
            return pts
        else:    
            vs = vertex_list(pts)
            return vs
    else:
        print('ERROR SURFACE IS NOT BSPLINE')

def pline_edge_frm_verts(vertex_list, attributes = {}):
    """
    This function constructs a polyline edge from a list of vertices.
 
    Parameters
    ----------
    vertex_list : a list of vertex
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    pline_edge : edge topology
        A edge topology containing a polyline geometry
    """
    edge = topobj.Edge(attributes = attributes)
    edge.add_polyline_curve(vertex_list)
    return edge

def bspline_edge_frm_xyzs(ctrlpts_xyz, degree = 2, resolution = 0.01, attributes = {}):
    """
    This function constructs a bspline edge from a list of control points.
 
    Parameters
    ----------
    ctrlpts_xyz : array of xyzs
        An array of xyzs. [[x1,y1,z1], [x2,y2,z2], [...]]
    
    degree : int, optional
        degree of the bspline curve. For straight lines use 2. For more curvy go for 3 and above
    
    resolution : float, optional 
        the resolution to store the curve. Float between 0-1. 0.1 will discretize the curve into 10 parts. 0.01 will discretize it to 100 parts. The lower the number the higher resolution of the curve stored in the edge topology. 
        
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    bspline_edge : edge topology
        A edge topology containing a bspline geometry
    """
    crv = BSpline.Curve()
    # Set degree
    crv.degree = degree
    
    # Set control points
    crv.ctrlpts = ctrlpts_xyz
    # Set knot vector
    crv.knotvector = utilities.generate_knot_vector(crv.degree, len(crv.ctrlpts))
    crv.delta = resolution
    
    edge = topobj.Edge(attributes = attributes)
    edge.add_bspline_curve(crv)
    return edge

def wire_frm_vertices(vertex_list, attributes = {}):
    """
    Constructs a wire from a list of vertices. Wire created from this will have edges containing lines (polylines with only 2 vertices, a straight line).
 
    Parameters
    ----------
    vertex_list : a list of vertices
        A list of vertex topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    wire : wire topology
        A wire topology 
    """
    edge_list = []
    n_v = len(vertex_list)
    for cnt,v in enumerate(vertex_list):
        if cnt != n_v-1:
            edge = pline_edge_frm_verts([v, vertex_list[cnt+1]])
        else:
            edge = pline_edge_frm_verts([v, vertex_list[0]])
        edge_list.append(edge)
    
    wire = wire_frm_edges(edge_list, attributes = attributes)
    
    return wire

def wire_frm_edges(edge_list, attributes = {}):
    """
    This function constructs a wire from a list of edges.
 
    Parameters
    ----------
    edge_list : a list of edge
        A list of edge topology
    
    attributes : dictionary, optional
        dictionary of the attributes.
 
    Returns
    -------
    wire : wire topology
        A wire topology 
    """
    wire = topobj.Wire(edge_list, attributes = attributes)
    return wire
    
def coordinate_system(origin, x_dir, y_dir):
    """
    This function creates a coordinate system object.
 
    Parameters
    ----------
    origin : tuple
        xyz coordinate that defines the origin.
    
    x_dir : tuple
        The xyz of a vector defining the x-axis
        
    y_dir : tuple
        The xyz of a vector defining the y-axis  
 
    Returns
    -------
    cs : the coordinate system object
        The coordinate system object.
    """
    return utility.CoordinateSystem(origin, x_dir, y_dir)

def composite(topo_list, attributes={}):
    """
    This function add attributes to the list of topology.
 
    Parameters
    ----------
    topo_list : Topo Object List
        List of Topo objects include Vertex, Edge, Wire, Face, Shell, Solid and Composite Topology Object.
        
    attributes : dictionary
        Dictionary of the attributes
 
    Returns
    -------
    composite : composite topology
        A composite topology containing the topo list
    """
    return topobj.Composite(topo_list, attributes = attributes)
    
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
    pass
    # from scipy import spatial
    # #TODO:need to think about the inheritance of the vertices, edges and wires 
    # srf_type = face.surface_type
    # if srf_type == geom.SrfType.POLYGON:
    #     nrml = face.surface.normal
    #     if nrml != np.array([0,0,1]) or nrml != np.array([0,0,-1]):
    #         #it must be transformed to be flat
    #         pass
        
    #     bdry_wire = face.bdry_wire
    #     bdry_verts = get.vertices_frm_wire(bdry_wire)
    #     xyz_list = np.array([v.point.xyz for v in bdry_verts])
        
    #     hole_wire_list = face.hole_wire_list
        
    #     hole_verts = []
    #     hole_xyz_list = np.array([pt.xyz for hole in hole_list for 
    #                               pt in hole])
    
    #     #TODO transform the points from 3d to 2d
    #     xy_list = np.delete(xyz_list, 2, axis=1)
    #     hole_xy_list = np.delete(hole_xyz_list, 2, axis=1)
        
    #     d_xy_list = np.concatenate((xy_list, hole_xy_list))
    #     tri = spatial.Delaunay(d_xy_list)
        
    #     chosen = d_xy_list[tri.simplices]
    #     # print(chosen)
    #     # for indices in tri.simplices:
    #     #     fp = d_xy_list[indices]
    #     #     print(fp)
    #         # create.polygon_face_frm_verts(vertex_list)
    #         # pt1 = list(xyz[verts[0]])
    #         # pt2 = list(xyz[verts[1]])
    #         # pt3 = list(xyz[verts[2]])
    #         # occtriangle = make_polygon([pt1,pt2,pt3])
    #         # tri_area = calculate.face_area(occtriangle)
    #         # if tri_area > tolerance:
    #         #     occtriangles.append(occtriangle)
#!python
# cython: language_level=3, c_string_type=str, c_string_encoding=ascii, boundscheck=False, wraparound=False, profile=False

import os 

import numpy as np
cimport numpy as np
import nibabel as nib
# from nibabel.affines import apply_affine
from scipy.linalg import inv

from dicelib.lazytractogram cimport LazyTractogram
from . import ui
from libc.math cimport sqrt
from libc.math cimport round as cround
from libcpp cimport bool



cdef float [:,::1] apply_affine(float [:,::1] end_pts, float [::1,:] M,
                                float [:] abc, float [:,::1] end_pts_trans) noexcept nogil:

    end_pts_trans[0][0] = ((end_pts[0][0]*M[0,0] + end_pts[0][1]*M[1,0] + end_pts[0][2]*M[2,0]) + abc[0])
    end_pts_trans[0][1] = ((end_pts[0][0]*M[0,1] + end_pts[0][1]*M[1,1] + end_pts[0][2]*M[2,1]) + abc[1])
    end_pts_trans[0][2] = ((end_pts[0][0]*M[0,2] + end_pts[0][1]*M[1,2] + end_pts[0][2]*M[2,2]) + abc[2])
    end_pts_trans[1][0] = ((end_pts[1][0]*M[0,0] + end_pts[1][1]*M[1,0] + end_pts[1][2]*M[2,0]) + abc[0])
    end_pts_trans[1][1] = ((end_pts[1][0]*M[0,1] + end_pts[1][1]*M[1,1] + end_pts[1][2]*M[2,1]) + abc[1])
    end_pts_trans[1][2] = ((end_pts[1][0]*M[0,2] + end_pts[1][1]*M[1,2] + end_pts[1][2]*M[2,2]) + abc[2])


    return end_pts_trans


cdef compute_grid( float thr, float[:] vox_dim ) :

    """ Compute the offsets grid
        Parameters
        ---------------------
        thr : double
            Radius of the radial search
            
        vox_dim : 1x3 numpy array
            Voxel dimensions
    """

    cdef float grid_center[3]
    cdef int thr_grid = <int> np.ceil(thr)

    # grid center
    cdef float x = 0
    cdef float y = 0
    cdef float z = 0
    cdef float[:,::1] centers_c
    cdef long[:] dist_grid

    grid_center[:] = [ x, y, z ]

    # create the mesh    
    mesh = np.linspace( -thr_grid, thr_grid, 2*thr_grid +1 )
    mx, my, mz = np.meshgrid( mesh, mesh, mesh )

    # find the centers of each voxels
    centers = np.stack([mx.ravel() + x, my.ravel() + y, mz.ravel() + z], axis=1)

    # sort the centers based on their distance from grid_center 
    dist_grid = ((centers - grid_center)**2).sum(axis=1).argsort()
    centers_c = centers[ dist_grid ].astype(np.float32)

    return centers_c




cpdef float [:,::1] to_matrix( float[:,::1] streamline, int n, float [:,::1] end_pts ) noexcept nogil:

    """ Retrieve the coordinates of the streamlines' endpoints.
    
    Parameters
    -----------------
    streamline: Nx3 numpy array
        The streamline data
        
    n: int
        Writes first n points of the streamline. If n<0 (default), writes all points.

    """
 
    cdef float *ptr = &streamline[0,0]
    cdef float *ptr_end = ptr+n*3-3

    end_pts[0,0]=ptr[0]
    end_pts[0,1]=ptr[1]
    end_pts[0,2]=ptr[2]
    end_pts[1,0]=ptr_end[0]
    end_pts[1,1]=ptr_end[1]
    end_pts[1,2]=ptr_end[2]

    return end_pts


cdef int[:] streamline_assignment_endpoints( int[:] start_vox, int[:] end_vox, int [:] roi_ret, float [:,::1] mat, int[:,:,::1] gm_v) noexcept nogil:

    cdef float [:] starting_pt = mat[0]
    cdef float [:] ending_pt = mat[1]
    start_vox[0]    = <int> cround(starting_pt[0])
    start_vox[1]    = <int> cround(starting_pt[1])
    start_vox[2]    = <int> cround(starting_pt[2])
    end_vox[0]      = <int> cround(ending_pt[0])
    end_vox[1]      = <int> cround(ending_pt[1])
    end_vox[2]      = <int> cround(ending_pt[2])

    roi_ret[0] = gm_v[ start_vox[0], start_vox[1], start_vox[2]]
    roi_ret[1] = gm_v[ end_vox[0], end_vox[1], end_vox[2]]
    return roi_ret


cdef int[:] streamline_assignment( float [:] start_pt_grid, int[:] start_vox, float [:] end_pt_grid, int[:] end_vox, int [:] roi_ret, float [:,::1] mat, float [:,::1] grid,
                            int[:,:,::1] gm_v, float thr, int[:] count_neighbours) noexcept nogil:

    """ Compute the label assigned to each streamline endpoint and then returns a list of connected regions.

    Parameters
    --------------
    start_pt_grid : 1x3 numpy array
        Starting point of the streamline in the grid space.
    start_vox : 1x3 numpy array
        Starting point of the streamline in the voxel space.
    end_pt_grid : 1x3 numpy array
        Ending point of the streamline in the grid space.
    end_vox : 1x3 numpy array
        Ending point of the streamline in the voxel space.
    roi_ret : 1x2 numpy array
        Labels assigned to the streamline endpoints.
    mat : 2x3 numpy array
        Streamline endpoints.
    grid : Nx3 numpy array
        Grid of voxels to check.
    gm_v : 3D numpy array
        GM map.
    thr : float
        Threshold used to compute the grid of voxels to check.
    """

    cdef float dist_s = 0
    cdef float dist_e = 0
    cdef size_t i = 0
    cdef int idx_s_min = 0
    cdef int idx_e_min = 0
    cdef float dist_s_temp = 1000
    cdef float dist_e_temp = 1000
    cdef int layer = 0

    roi_ret[0] = 0
    roi_ret[1] = 0

    cdef float [:] starting_pt = mat[0]
    cdef float [:] ending_pt = mat[1]
    cdef int grid_size = grid.shape[0]

    for i in xrange(grid_size):
        # from 3D coordinates to index
        start_pt_grid[0] = starting_pt[0] + grid[i][0]
        start_pt_grid[1] = starting_pt[1] + grid[i][1]
        start_pt_grid[2] = starting_pt[2] + grid[i][2]

        # check if the voxel is inside the mask
        if cround(start_pt_grid[0]) < 0 or cround(start_pt_grid[0]) >= gm_v.shape[0] or cround(start_pt_grid[1]) < 0 or cround(start_pt_grid[1]) >= gm_v.shape[1] or cround(start_pt_grid[2]) < 0 or cround(start_pt_grid[2]) >= gm_v.shape[2]:
            continue

        start_vox[0] = <int> cround(start_pt_grid[0])
        start_vox[1] = <int> cround(start_pt_grid[1])
        start_vox[2] = <int> cround(start_pt_grid[2])

        if gm_v[ start_vox[0], start_vox[1], start_vox[2]] > 0:
            dist_s = sqrt( ( starting_pt[0] - (<int>start_pt_grid[0] + 0.5) )**2 + ( starting_pt[1] - (<int>start_pt_grid[1] + 0.5) )**2 + ( starting_pt[2] - (<int>start_pt_grid[2] + 0.5) )**2 )
            if dist_s <= thr and dist_s < dist_s_temp:
                roi_ret[0] = gm_v[ start_vox[0], start_vox[1], start_vox[2]]
                dist_s_temp = dist_s
        if i == count_neighbours[layer]:
            if dist_s_temp < 1000:
                break
            else:
                layer += 1
    layer = 0
    for i in xrange(grid_size):
        end_pt_grid[0] = ending_pt[0] + grid[i][0]
        end_pt_grid[1] = ending_pt[1] + grid[i][1]
        end_pt_grid[2] = ending_pt[2] + grid[i][2]

        if cround(end_pt_grid[0]) < 0 or cround(end_pt_grid[0]) >= gm_v.shape[0] or cround(end_pt_grid[1]) < 0 or cround(end_pt_grid[1]) >= gm_v.shape[1] or cround(end_pt_grid[2]) < 0 or cround(end_pt_grid[2]) >= gm_v.shape[2]:
            continue

        end_vox[0] = <int> cround(end_pt_grid[0])
        end_vox[1] = <int> cround(end_pt_grid[1])
        end_vox[2] = <int> cround(end_pt_grid[2])

        if gm_v[ end_vox[0], end_vox[1], end_vox[2]  ] > 0:
            dist_e = sqrt( ( ending_pt[0] - (<int>end_pt_grid[0] + 0.5) )**2 + ( ending_pt[1] - (<int>end_pt_grid[1] + 0.5) )**2 + ( ending_pt[2] - (<int>end_pt_grid[2] + 0.5) )**2 )

            if dist_e <= thr and dist_e < dist_e_temp:
                roi_ret[1] = gm_v[ end_vox[0], end_vox[1], end_vox[2]]
                dist_e_temp = dist_e
        if i == count_neighbours[layer]:
            if dist_e_temp < 1000:
                break
            else:
                layer += 1

    return roi_ret


cpdef assign( input_tractogram: str, int[:] pbar_array, int id_chunk, int start_chunk, int end_chunk, gm_map_file: str, threshold: 2 ):

    """ Compute the assignments of the streamlines based on a GM map.
    
    Parameters
    ----------
    input_tractogram : string
        Path to the file (.tck) containing the streamlines to process.
    pbar_array : numpy array
        Array of integers used to update the progress bar.
    id_chunk : int
        Index of the chunk.
    start_chunk : int
        Index of the first streamline of the chunk.
    end_chunk : int
        Index of the last streamline of the chunk.
    gm_map_file : string
        Path to the GM map file.
    threshold : int
        Threshold used to compute the grid of voxels to check.
    verbose : bool
        If True, display information about the processing.
    """


    if not os.path.isfile(input_tractogram):
        ui.ERROR( f'File "{input_tractogram}" not found' )
    if not os.path.isfile(gm_map_file):
        ui.ERROR( f'File "{gm_map_file}" not found' )

    
    # Load of the gm map
    gm_map_img = nib.load(gm_map_file)
    gm_map_data = gm_map_img.get_fdata()
    ref_data = gm_map_img
    ref_header = ref_data.header
    affine = ref_data.affine
    cdef int [:,:,::1] gm_map = np.ascontiguousarray(gm_map_data, dtype=np.int32)

    cdef float [:,::1] inverse = np.ascontiguousarray(inv(affine), dtype=np.float32) #inverse of affine
    cdef float [::1,:] M = inverse[:3, :3].T 
    cdef float [:] abc = inverse[:3, 3]
    cdef float [:] voxdims = np.asarray( ref_header.get_zooms(), dtype = np.float32 )

    cdef float thr = <float> threshold/np.max(voxdims)
    cdef float [:,::1] grid
    cdef size_t i = 0  
    cdef int n_streamlines = end_chunk - start_chunk
    cdef float [:,::1] matrix = np.zeros( (2,3), dtype=np.float32)
    assignments = np.zeros( (n_streamlines, 2), dtype=np.int32 )
    cdef int[:,:] assignments_view = assignments

    cdef float [:,::1] end_pts = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] end_pts_temp = np.zeros((2,3), dtype=np.float32)
    cdef float [:,::1] end_pts_trans = np.zeros((2,3), dtype=np.float32)
    cdef float [:] start_pt_grid = np.zeros(3, dtype=np.float32)
    cdef int [:] start_vox = np.zeros(3, dtype=np.int32)
    cdef float [:] end_pt_grid = np.zeros(3, dtype=np.float32)
    cdef int [:] end_vox = np.zeros(3, dtype=np.int32)
    cdef int [:] roi_ret = np.array([0,0], dtype=np.int32)

    TCK_in = None
    TCK_in = LazyTractogram( input_tractogram, mode='r' )
    # compute the grid of voxels to check
    grid = compute_grid( thr, voxdims )
    layers = np.arange( 1,<int> np.ceil(thr)+1, 2 )
    neighbs = [v**3-1 for v in layers]
    cdef int[:] count_neighbours = np.array(neighbs, dtype=np.int32)

    if thr < 0.5 :
        with nogil:
            while i < start_chunk:
                TCK_in._read_streamline()
                i += 1
            for i in xrange( n_streamlines ):
                TCK_in._read_streamline()
                end_pts = to_matrix( TCK_in.streamline, TCK_in.n_pts, end_pts_temp )
                matrix = apply_affine(end_pts, M, abc, end_pts_trans)
                assignments_view[i] = streamline_assignment_endpoints( start_vox, end_vox, roi_ret, matrix, gm_map)
                pbar_array[id_chunk] += 1

    else:
        thr += 0.5
        with nogil:
            while i < start_chunk:
                TCK_in._read_streamline()
                i += 1
            for i in xrange( n_streamlines ):
                TCK_in._read_streamline()
                end_pts = to_matrix( TCK_in.streamline, TCK_in.n_pts, end_pts_temp )
                matrix = apply_affine(end_pts, M, abc, end_pts_trans)
                assignments_view[i] = streamline_assignment( start_pt_grid, start_vox, end_pt_grid, end_vox, roi_ret,
                                                            matrix, grid, gm_map, thr, count_neighbours)
                pbar_array[id_chunk] += 1


    if TCK_in is not None:
        TCK_in.close()
    return assignments